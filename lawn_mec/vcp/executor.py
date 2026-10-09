from copy import deepcopy
import numpy as np
from lawn_mec.exec_v2.events import EventExecutor
from lawn_mec.exec_v2.motion import Target
from lawn_mec.exec_v2.policies import estimated_readiness

VARIANTS = ('earliest_feasible', 'bin_start')


class SuffixExecutor(EventExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncated = set()

    def clone(self):
        other = super().clone()
        other.truncated = self.truncated.copy()
        return other

    def truncation_options(self, u):
        node, start = self._canonical(u); ell = self.layers[u]
        routes = []
        for route, path in enumerate(self.planner.paths(node, self.final_nodes[u])):
            travel = sum(min(p.duration for p in self.lib.straight(a, b, self.lib.heights[ell])) for a, b in zip(path[:-1], path[1:]))
            legal = start+travel <= self.planner.landing_start
            energy = None
            if legal:
                plan = self.planner.plan(self.lib.point(node, self.lib.heights[ell]), np.zeros(3), start, Target('endpoint', uav=u), self.state.N, self.lib.heights[ell], route_path=path)
                legal, energy = plan.status == 'ok', plan.energy
            routes.append(dict(route=route, path=list(path), earliest=self.state.N, latest=self.state.N, legal=legal, energy_J=energy))
        return routes

    def _apply(self, event, decisions):
        if event['kind'] != 'commitments':
            return super()._apply(event, decisions)
        event, decisions = deepcopy(event), deepcopy(decisions)
        for req in event['requests']:
            u = req['uav']; choice = decisions.get('commitments', {}).get(u, {})
            if 'truncate' not in choice:
                continue
            if choice.pop('truncate') is not True or req['endpoint']:
                raise ValueError('truncate requires an unfinished visit suffix')
            routes = self.truncation_options(u)
            req.update(endpoint=True, visit=None, routes=routes, earliest=self.state.N, latest=self.state.N, keep=False)
            self.truncated.add(u)
        return super()._apply(event, decisions)

    def result(self):
        result = super().result()
        result['truncated_uavs'] = sorted(self.truncated)
        return result


class CommitmentPolicy:
    def __init__(self, commitment, tables, variant='earliest_feasible'):
        if variant not in VARIANTS:
            raise ValueError('unknown bin mapping')
        self.commitment, self.tables, self.variant = commitment, tables, variant
        self.deviations = []

    def visit_target(self, interval, req):
        left, right = interval
        low, high = max(left, req['earliest']), min(right, req['latest'])
        if self.variant == 'earliest_feasible' and low <= high:
            return low
        return min(max(left, req['earliest']), req['latest'])

    def priorities(self, ex, event):
        bands = {'X': (12, 16), 'Y': (6, 11), 'Z': (1, 5)}
        K = len(ex.state.tasks); result = np.zeros((K, 3))
        ranked = sorted(range(K), key=lambda k: (ex.state.tasks[k]['d'], k))
        ranks = {k: i for i, k in enumerate(ranked)}
        for k, stage in event['active']:
            c = self.commitment.tasks[k]
            low, high = bands[c.priority]
            inside = c.upload is not None and self.tables.upload_bins[k][c.upload][0] <= ex.state.n <= self.tables.upload_bins[k][c.upload][1]
            level = high if inside else low
            result[k, stage] = (level+(K-ranks[k])/(2*(K+1)))/16
        return result.tolist()

    def decide(self, ex, event):
        kind = event['kind']
        if kind == 'layers':
            return dict(layers=list(self.commitment.layers))
        if kind == 'locations':
            choices = {}
            for k in event['tasks']:
                c = self.commitment.tasks[k]
                choices[k] = c.location if c.g == 'S' else min(event['legal_locations'][k], key=lambda m: (estimated_readiness(ex, k, m), m))
            return dict(locations=choices)
        if kind == 'priorities':
            return dict(priorities=self.priorities(ex, event))
        if kind != 'commitments':
            raise ValueError('no decision at done')
        choices = {}
        for req in event['requests']:
            u = req['uav']; seq = ex.state.instance['visits'][u]
            remaining = seq[req['pointer']:]
            if remaining and all(self.commitment.tasks[k].g == 'G' for k, _ in remaining):
                routes = [r for r in ex.truncation_options(u) if r['legal']]
                if not routes:
                    raise ValueError('suffix termination has no legal return')
                r = min(routes, key=lambda r: (r['energy_J'], r['route']))
                choices[u] = dict(truncate=True, T=ex.state.N, route=r['route'])
                continue
            if req['reason'] == 'deferral' and req['keep']:
                choices[u] = dict(T=ex.state.n, route='keep'); continue
            legal = [r for r in req['routes'] if r['legal']]
            node, start = ex._canonical(u); h = ex.lib.heights[ex.layers[u]]
            if req['endpoint']:
                target, T = Target('endpoint', uav=u), ex.state.N
            else:
                k, phase = req['visit']; task = ex.state.tasks[k]; c = self.commitment.tasks[k]
                target = Target('service', task=k, phase=phase)
                if c.g == 'G':
                    kernel = self.tables.travel_energy(node, task['z_'+phase], ex.layers[u])
                    lo, hi = req['earliest']-start, req['latest']-start
                    T = start+lo+int(np.argmin(kernel[lo:hi+1]))
                else:
                    b = c.s if phase == 's' else c.a
                    left, right = self.tables.visit_bins[k, phase][b]
                    T = self.visit_target((left, right), req)
                    if not left <= T <= right:
                        self.deviations.append(dict(slot=ex.state.n, uav=u, task=k, phase=phase, bin=b, target=T, reason='bin_unreachable_nearest_slot'))
            candidates = []
            for r in legal:
                if r['earliest'] <= T <= r['latest']:
                    plan = ex.planner.plan(ex.lib.point(node, h), np.zeros(3), start, target, int(T), h, route_path=r['path'])
                    if plan.status == 'ok':
                        candidates.append((plan.energy, r['route']))
            if not candidates:
                raise ValueError('no route at explicitly selected target')
            choices[u] = dict(T=int(T), route=min(candidates)[1])
        return dict(commitments=choices)


def rollout(instance, commitment, tables, variant='earliest_feasible', *, executor=None, capture=False,
            execution_layer='evaluation'):
    if execution_layer not in ('evaluation', 'audit'):
        raise ValueError('execution_layer must be evaluation or audit')
    ex = executor or SuffixExecutor(instance, library=tables.lib, planner=tables.planner,
                                   execution_layer=execution_layer)
    policy = CommitmentPolicy(commitment, tables, variant)
    snapshots = []
    while True:
        event = ex.next_event()
        if event['kind'] == 'done':
            result = ex.result()
            result.update(bin_variant=variant, deviations=policy.deviations)
            return (result, snapshots) if capture else result
        decision = policy.decide(ex, event)
        if capture:
            snapshots.append((ex.clone(), event, decision))
        ex.apply(decision)


def fork_rollout(instance, commitment, tables, variant, snapshots):
    policy = CommitmentPolicy(commitment, tables, variant)
    for index, (ex, event, old_decision) in enumerate(snapshots):
        prior_deviations = list(policy.deviations)
        if policy.decide(ex, event) != old_decision:
            result, tail = rollout(instance, commitment, tables, variant, executor=ex.clone(), capture=True)
            result['fork_slot'] = ex.state.n
            result['deviations'] = prior_deviations+result['deviations']
            return result, snapshots[:index]+tail
    ex = snapshots[-1][0].clone()
    ex.apply(snapshots[-1][2])
    result = ex.result(); result.update(bin_variant=variant, deviations=policy.deviations, fork_slot=ex.state.N)
    return result, snapshots
