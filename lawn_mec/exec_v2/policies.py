from copy import deepcopy
import math
from fractions import Fraction
import numpy as np
from lawn_mec.env.model import motion, slot_capacity
from .allocator import boundary_tasks
from .events import EventExecutor
from .primitives import Uncertified

METHODS = ('P_hand-EL', 'all-local-EL', 'all-offload-EL')


def slack_priorities(executor):
    state = executor.state
    w = [1./(max(0., t['d']-state.n*state.p['delta'])+1.) for t in state.tasks]
    return np.repeat(np.array(w)[:, None], 3, axis=1)


def forecast_motion(executor, u):
    s, lib = executor.state, executor.lib
    ell, n = executor.layers[u], s.n
    h = lib.heights[ell]
    rows = list(executor.schedules[u][n:])
    c = executor.commitments[u]
    if c is None:
        node, begin = executor._canonical(u)
        pointer = s.visit_pointer[u]
    elif c['endpoint']:
        pointer = len(s.instance['visits'][u]); node = executor.final_nodes[u]
    else:
        k, phase = s.instance['visits'][u][c['pointer']]
        node = s.tasks[k]['z_'+phase]; pointer = c['pointer']+1
    if c is None or not c['endpoint']:
        chain = []
        for k, phase in s.instance['visits'][u][pointer:]:
            dest = s.tasks[k]['z_'+phase]
            paths = executor.planner.paths(node, dest)
            legs = min((tuple(min(lib.straight(a, b, h), key=lambda p: (p.duration, p.E, p.name))
                              for a, b in zip(path[:-1], path[1:])) for path in paths),
                       key=lambda ps: (sum(p.duration for p in ps), sum(p.E for p in ps)))
            chain.extend(legs)
            chain.append(min(lib.service(k, phase, h), key=lambda p: (p.E, p.name)))
            node = dest
        rows.extend(executor._rows(chain))
    q, v = s.q[u].copy(), s.v[u].copy()
    trajectory = []
    for offset in range(s.N-n):
        a = np.asarray(rows[offset]['accel']) if offset < len(rows) else np.zeros(3)
        trajectory.append((q.copy(), v.copy(), a.copy()))
        q, v = motion(q, v, a, s.p['delta'])
    return trajectory


def estimated_readiness(executor, k, location=None, *, trajectory=None, release_slot=None):
    s = executor.state; task, st, p = s.tasks[k], s.states[k], s.p
    service_st = (st if executor.service is None else
                  executor.service.task_view(k, task, st, s.n))
    if service_st.events['r'].available:
        return service_st.events['r'].value
    loc = executor.locations[k] if location is None else location
    if loc is None:
        raise ValueError('readiness estimate needs an explicit prospective location')
    locations = executor.locations.copy()
    pending = []
    from .state import clone_task
    from lawn_mec.env.queue import inject
    for j, other in enumerate(s.states):
        cp = clone_task(other)
        cp.location = loc if j == k else locations[j]
        if cp.location is not None:
            inject(s.tasks[j], cp, s.n)
        if executor.service is not None:
            cp = executor.service.task_view(j, s.tasks[j], cp, s.n)
        pending.append(cp)
    target = pending[k]
    W = target.W.copy()
    if not st.events['g'].available:
        W[1 if loc == 0 else 0] = task['C'] if loc == 0 else task['D']
    priorities = slack_priorities(executor)[:, 1]
    key = lambda j: (-priorities[j]/s.tasks[j]['C'], j)
    ahead = sum(other.W[1] for j, other in enumerate(pending)
                if j != k and other.location == loc and (loc != 0 or s.tasks[j]['u'] == task['u'])
                and key(j) < key(k))
    if s.layer == 'audit':
        W = [Fraction(float(x)) for x in W]
        ahead = sum((Fraction(float(other.W[1])) for j, other in enumerate(pending)
                     if j != k and other.location == loc and (loc != 0 or s.tasks[j]['u'] == task['u'])
                     and key(j) < key(k)), Fraction(0))
    release = s.n if release_slot is None else int(release_slot)
    if release < s.n:
        raise ValueError('prospective release precedes current slot')
    if loc == 0:
        if s.layer == 'audit':
            capacity = Fraction(p['delta'])*Fraction(p['F_local'])
            ahead = max(Fraction(0), ahead-(release-s.n)*capacity)
            return min(s.N+1, release+math.ceil((ahead+W[1])/capacity))
        ahead = max(0., ahead-(release-s.n)*p['delta']*p['F_local'])
        return min(s.N+1, release+math.ceil((ahead+W[1])/(p['delta']*p['F_local'])))
    if s.layer == 'audit':
        ahead = max(Fraction(0), ahead-sum((Fraction(p['delta'])*Fraction(
            p['G_total']-s.instance['load'][loc-1][n])
            for n in range(s.n, min(release, s.N))), Fraction(0)))
    else:
        ahead = max(0., ahead-sum(p['delta']*(p['G_total']-s.instance['load'][loc-1][n])
                                 for n in range(s.n, min(release, s.N))))
    trajectory = trajectory if trajectory is not None else forecast_motion(executor, task['u'])
    if W[2] > 0 or target.events['c'].available:
        stage = 2
    elif W[1] > 0 or target.events['ul'].available:
        stage = 1
    else:
        stage = 0
    for offset, (q, v, a) in enumerate(trajectory):
        n = s.n+offset
        if n < release:
            continue
        if stage == 1:
            cap = (Fraction(p['delta'])*Fraction(p['G_total']-s.instance['load'][loc-1][n])
                   if s.layer == 'audit' else p['delta']*(p['G_total']-s.instance['load'][loc-1][n]))
            used = min(ahead, cap); ahead -= used; cap -= used
        else:
            if s.layer == 'audit':
                cap = Fraction(s.physics.radio(q, v, a, loc-1,
                                               p['P_ul'] if stage == 0 else p['P_dl'], 1.).lo)
            else:
                cap = slot_capacity(q, v, a, s.instance['bs'][loc-1], s.instance['buildings'],
                                    p['P_ul'] if stage == 0 else p['P_dl'], 1., p, 'guard', s.cfg)
        W[stage] -= cap
        if W[stage] <= 0:
            if stage == 2:
                return n+1
            stage += 1; W[stage] = task['C'] if stage == 1 else task['O']
            if s.layer == 'audit':
                W[stage] = Fraction(W[stage])
    return s.N+1


class HandPolicy:
    def __init__(self, method='P_hand-EL', plan=None):
        if method not in METHODS:
            raise ValueError('unknown EL rule policy')
        self.method, self.plan = method, deepcopy(plan or {})

    def decide(self, executor, event):
        s = executor.state; kind = event['kind']
        if kind == 'layers':
            if 'layers' in self.plan:
                return dict(layers=self.plan['layers'])
            layers = []
            for mask in event['masks']:
                options = [i for i, legal in enumerate(mask) if legal and i not in layers]
                if not options:
                    raise Uncertified('indexed-layer policy has no unused feasible layer')
                layers.append(min(options))
            return dict(layers=layers)
        if kind == 'locations':
            result = {}
            for k in event['tasks']:
                if str(k) in self.plan.get('locations', {}):
                    result[k] = self.plan['locations'][str(k)]; continue
                if self.method == 'all-local-EL':
                    result[k] = 0; continue
                trajectory = forecast_motion(executor, s.tasks[k]['u'])
                options = range(1 if self.method == 'all-offload-EL' else 0, s.p['M']+1)
                result[k] = min(options, key=lambda m: (estimated_readiness(executor, k, m, trajectory=trajectory), m))
            return dict(locations=result)
        if kind == 'priorities':
            return dict(priorities=slack_priorities(executor).tolist())
        if kind != 'commitments':
            raise ValueError('no decision for this event')
        result = {}
        for req in event['requests']:
            u, pointer = req['uav'], req['pointer']; key = f'{u}:{pointer}'
            legal = [r for r in req['routes'] if r['legal']]
            route = self.plan.get('routes', {}).get(key)
            if route is None:
                r = min(legal, key=lambda r: (r['energy_J'], r['route']))
                route = r['route']
            else:
                r = next((r for r in legal if r['route'] == route), None)
                if r is None:
                    raise Uncertified(f'planned route {key}={route} is masked')
            T = r['earliest']
            if key in self.plan.get('offsets', {}):
                if req['reason'] == 'deferral':
                    result[u] = dict(T=s.n, route='keep'); continue
                T += self.plan['offsets'][key]
            elif not req['endpoint']:
                k, phase = req['visit']; task = s.tasks[k]
                if phase == 'a':
                    ready = estimated_readiness(executor, k)
                    ns = s.states[k].events['s'].value
                    deadline = math.floor(task['d']/s.p['delta'])-task['L_a']
                    fresh = ns+math.floor(task['H']/s.p['delta']) if ns is not None else -1
                    candidate = max(T, ready)
                    if candidate <= min(r['latest'], deadline, fresh):
                        T = candidate
                else:
                    span, node = task['L_s'], task['z_s']
                    h = executor.lib.heights[executor.layers[u]]
                    for other_k, other_phase in s.instance['visits'][u][pointer+1:]:
                        other = s.tasks[other_k]; dest = other['z_'+other_phase]
                        span += executor.planner.earliest_duration(node, dest, h)
                        if (other_k, other_phase) == (k, 'a'):
                            break
                        span += other['L_'+other_phase]; node = dest
                    deadline = math.floor(task['d']/s.p['delta'])-task['L_a']
                    H = math.floor(task['H']/s.p['delta'])
                    if span <= H and T+span <= deadline:
                        for proposed in range(T, min(r['latest'], deadline-span)+1):
                            virtual = executor.clone()
                            one = dict(kind='commitments', requests=[req])
                            virtual._apply(one, dict(commitments={u: dict(T=proposed, route=route)}))
                            trajectory = forecast_motion(virtual, u)
                            locations = (range(1, s.p['M']+1) if self.method == 'all-offload-EL' else
                                         (0,) if self.method == 'all-local-EL' else range(s.p['M']+1))
                            ready = min(estimated_readiness(virtual, k, m, trajectory=trajectory,
                                          release_slot=proposed+task['L_s']) for m in locations)
                            if max(proposed+span, ready) <= min(proposed+H, deadline):
                                T = proposed; break
            if not r['earliest'] <= T <= r['latest']:
                raise Uncertified(f'planned target {key}={T} is masked')
            if key not in self.plan.get('routes', {}) and req['reason'] != 'deferral':
                node, start = executor._canonical(u); h = executor.lib.heights[executor.layers[u]]
                from .motion import Target
                target = (Target('endpoint', uav=u) if req['endpoint'] else
                          Target('service', task=req['visit'][0], phase=req['visit'][1]))
                options = []
                for candidate in legal:
                    if candidate['earliest'] <= T <= candidate['latest']:
                        motion = executor.planner.plan(executor.lib.point(node, h), np.zeros(3), start,
                                 target, T, h, route_path=candidate['path'])
                        if motion.status == 'ok':
                            options.append((motion.energy, candidate['route']))
                route = min(options)[1]
            if req['reason'] == 'deferral' and T == s.n and req['keep']:
                result[u] = dict(T=T, route='keep')
            else:
                result[u] = dict(T=int(T), route=int(route))
        return dict(commitments=result)


def rollout(instance, method='P_hand-EL', plan=None, *, library=None, planner=None, certified_service=True,
            execution_layer='evaluation'):
    executor = EventExecutor(instance, library=library, planner=planner, certified_service=certified_service,
                             execution_layer=execution_layer)
    policy = HandPolicy(method, plan)
    while True:
        event = executor.next_event()
        if event['kind'] == 'done':
            return executor.result()
        executor.apply(policy.decide(executor, event))


def episode_plan(result):
    plan = dict(layers=None, offsets={}, routes={}, locations={})
    earliest = {}
    for event in result['decisions']:
        choices = event['choices']
        if event['kind'] == 'layers':
            plan['layers'] = list(choices['layers'])
        elif event['kind'] == 'locations':
            plan['locations'].update({str(k): v for k, v in choices['locations'].items()})
        elif event['kind'] == 'commitments':
            for req in event['requests']:
                u = req['uav']; key = f"{u}:{req['pointer']}"
                choice = choices['commitments'][u]
                if req['reason'] == 'deferral':
                    if key in earliest:
                        plan['offsets'][key] = choice['T']-earliest[key]
                    continue
                if choice['route'] == 'keep':
                    continue
                plan['routes'][key] = choice['route']
                earliest[key] = req['routes'][choice['route']]['earliest']
                plan['offsets'][key] = choice['T']-earliest[key]
    return plan
