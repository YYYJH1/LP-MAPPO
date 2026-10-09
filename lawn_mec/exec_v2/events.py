from copy import copy, deepcopy
from time import perf_counter
import numpy as np
from lawn_mec.env.checker import replay_episode
from .allocator import allocate, boundary_tasks
from .demo_rule import node_of
from .motion import MotionPlanner, Target
from .primitives import PrimitiveLibrary, Uncertified
from .state import SimState, freeze, thaw
from .step import SlotActions, step_v2
from .service import CertifiedService


class EventExecutor:
    def __init__(self, instance, *, library=None, planner=None, certified_service=True,
                 execution_layer='evaluation', terminal_replay=True):
        if not isinstance(certified_service, bool):
            raise ValueError('certified_service must be a boolean')
        if execution_layer == 'audit' and not certified_service:
            raise ValueError('audit execution requires certified_service=True')
        if not isinstance(terminal_replay, bool):
            raise ValueError('terminal_replay must be a boolean')
        if not terminal_replay and (execution_layer != 'audit' or not certified_service):
            raise ValueError('training without terminal replay requires audit execution and certified service')
        self.terminal_replay = terminal_replay
        self.state = SimState(instance, layer=execution_layer)
        self.service = CertifiedService(len(self.state.tasks)) if certified_service else None
        self.lib = library or PrimitiveLibrary(instance)
        self.planner = planner or MotionPlanner(self.lib, max_alternatives=3)
        if self.planner.max_alternatives != 3:
            raise ValueError('event interface requires four path candidates')
        if not hasattr(self.planner, 'minimum_times'):
            self.planner.precompute()
        U = self.state.p['U']
        if U > len(self.lib.heights):
            raise Uncertified('exclusive layer count exceeded')
        self.initial_nodes = [node_of(instance, q) for q in instance['qI']]
        self.final_nodes = [node_of(instance, q) for q in instance['qF']]
        self.takeoffs = {}
        for u in range(U):
            node = self.initial_nodes[u]
            if np.linalg.norm(self.state.v[u]) > self.lib.cfg['epsilon_num'] or np.linalg.norm(
                    self.state.q[u]-self.lib.point(node, 85.)) > self.lib.cfg['epsilon_num']:
                raise Uncertified('initial state must be a node centre at rest at 85 m')
            for ell, h in enumerate(self.lib.heights):
                self.takeoffs[u, ell] = min(self.lib.vertical(node, 85., h),
                                          key=lambda x: (x.duration, x.E, x.name))
        self.takeoff_end = max(p.duration for p in self.takeoffs.values())
        self.layers = None
        self.schedules = [()] * U
        self.commitments = [None] * U
        self.pending = set(range(U))
        self.locations = [None] * len(self.state.tasks)
        self.deferrals = set()
        self.approved_starts = set()
        self.decisions = []
        self.timings = dict(event_s=[], allocator_s=[], step_s=[])
        self._cached_event = None
        self._remaining_cache = {}

    def clone(self):
        other = copy(self)
        other.state = self.state.clone()
        other.service = None if self.service is None else self.service.clone()
        other.layers = None if self.layers is None else self.layers.copy()
        other.schedules = self.schedules.copy()
        other.commitments = deepcopy(self.commitments)
        other.pending, other.deferrals = self.pending.copy(), self.deferrals.copy()
        other.approved_starts = self.approved_starts.copy()
        other.locations, other.decisions = self.locations.copy(), self.decisions.copy()
        other.timings = {k: v.copy() for k, v in self.timings.items()}
        other._cached_event = deepcopy(self._cached_event)
        other._remaining_cache = self._remaining_cache.copy()
        return other

    def task_states(self):
        from .state import clone_task
        from lawn_mec.env.queue import inject
        pending = []
        for k, st in enumerate(self.state.states):
            cp = clone_task(st)
            cp.location = self.locations[k]
            if cp.location is not None:
                inject(self.state.tasks[k], cp, self.state.n)
            if self.service is not None:
                cp = self.service.task_view(k, self.state.tasks[k], cp, self.state.n)
            pending.append(cp)
        return pending

    def change_layer(self, uav, layer):
        raise NotImplementedError('R1\u2032: stage 3 supports exclusive layers at n=0 only')

    def remaining_time(self, u, pointer, node, ell):
        key = u, pointer, node, ell
        if key not in self._remaining_cache:
            h, total = self.lib.heights[ell], 0
            for k, phase in self.state.instance['visits'][u][pointer:]:
                t = self.state.tasks[k]; dest = t['z_'+phase]
                total += int(self.planner.minimum_times[ell, node, dest])+t['L_'+phase]
                node = dest
            total += int(self.planner.minimum_times[ell, node, self.final_nodes[u]])
            self._remaining_cache[key] = int(total)
        return self._remaining_cache[key]

    def _canonical(self, u):
        if self.state.n == 0:
            return self.initial_nodes[u], self.takeoff_end
        h = self.lib.heights[self.layers[u]]
        node = node_of(self.state.instance, self.state.q[u])
        if np.linalg.norm(self.state.q[u]-self.lib.point(node, h)) > self.lib.cfg['epsilon_num'] or np.linalg.norm(self.state.v[u]) > self.lib.cfg['epsilon_num']:
            raise Uncertified('commitment must occur at a certified rest centre')
        return node, self.state.n

    def commitment_options(self, u):
        node, start = self._canonical(u)
        ell = self.layers[u]; h = self.lib.heights[ell]
        pointer = self.state.visit_pointer[u]; visits = self.state.instance['visits'][u]
        endpoint = pointer == len(visits)
        if endpoint:
            target = Target('endpoint', uav=u); dest = self.final_nodes[u]
            latest, length = self.state.N, 0
        else:
            k, phase = visits[pointer]; task = self.state.tasks[k]
            target = Target('service', task=k, phase=phase); dest = task['z_'+phase]
            length = task['L_'+phase]
            latest = self.planner.landing_start-length-self.remaining_time(u, pointer+1, dest, ell)
        candidates = []
        for route, path in enumerate(self.planner.paths(node, dest)):
            travel = sum(min(p.duration for p in self.lib.straight(a, b, h))
                         for a, b in zip(path[:-1], path[1:]))
            earliest = self.state.N if endpoint else start+travel
            legal = (start+travel <= self.planner.landing_start) if endpoint else earliest <= latest
            energy = None
            if legal:
                plan = self.planner.plan(self.lib.point(node, h), np.zeros(3), start, target,
                                         earliest, h, route_path=path)
                legal = plan.status == 'ok'
                if legal:
                    energy = plan.energy
            candidates.append(dict(route=route, path=list(path), earliest=int(earliest),
                                   latest=int(latest), legal=bool(legal), energy_J=energy,
                                   travel_slots=int(travel), energy_at_slot=int(earliest)))
        legal = [r for r in candidates if r['legal']]
        if not legal:
            raise Uncertified(f'UAV {u} has no route/time pair that preserves all visits and landing')
        current = self.commitments[u]
        keep = bool(current is not None and current['pointer'] == pointer and current['T'] >= self.state.n)
        return dict(uav=u, pointer=pointer, visit=None if endpoint else list(visits[pointer]),
                    endpoint=endpoint, earliest=min(r['earliest'] for r in legal), latest=int(latest),
                    routes=candidates, keep=keep, keep_T=current['T'] if keep else None,
                    battery_mask=False, layer_change=False)

    def request_deferral(self, uav):
        c = self.commitments[uav]
        if c is None or c['T'] != self.state.n or c['endpoint']:
            raise ValueError('deferral requires the current visit start boundary')
        self.deferrals.add(uav); self._cached_event = None

    def next_event(self):
        if self._cached_event is not None:
            return deepcopy(self._cached_event)
        tic = perf_counter(); n = self.state.n
        if n == self.state.N:
            event = dict(kind='done', slot=n)
        elif self.layers is None:
            masks = [[self.takeoff_end+self.remaining_time(u, 0, self.initial_nodes[u], ell)
                      <= self.planner.landing_start for ell in range(len(self.lib.heights))]
                     for u in range(self.state.p['U'])]
            event = dict(kind='layers', slot=n, uavs=list(range(self.state.p['U'])),
                         heights=list(self.lib.heights), masks=masks, exclusive=True,
                         takeoff_end=self.takeoff_end)
        else:
            for u, c in enumerate(self.commitments):
                if c is None or c['endpoint'] or c['T'] != n or (u, c['pointer'], n) in self.approved_starts:
                    continue
                k, phase = self.state.instance['visits'][u][c['pointer']]
                ready = (self.state.states[k] if self.service is None else
                         self.service.task_view(k, self.state.tasks[k], self.state.states[k], n)).events['r'].value
                if phase == 'a' and (ready is None or ready > n):
                    self.deferrals.add(u)
            needing = sorted(self.pending | self.deferrals)
            ks = [k for k, st in enumerate(self.state.states)
                  if st.events['g'].value == n and self.locations[k] is None]
            if ks:
                event = dict(kind='locations', slot=n, tasks=ks,
                             legal_locations={k: list(range(self.state.p['M']+1)) for k in ks})
            elif needing:
                event = dict(kind='commitments', slot=n,
                             requests=[dict(**self.commitment_options(u),
                                            reason='deferral' if u in self.deferrals else 'initial' if n == 0 else 'visit_end')
                                       for u in needing])
            else:
                pending = boundary_tasks(self.state, self.locations, service=self.service)
                event = dict(kind='priorities', slot=n, shape=[len(pending), 3],
                             active=[[k, i] for k, st in enumerate(pending) for i in range(3) if st.W[i] > 0],
                             minimum=0.)
        self.timings['event_s'].append(perf_counter()-tic)
        self._cached_event = deepcopy(event)
        return event

    @staticmethod
    def _rows(chain):
        return tuple(freeze(dict(edge=e, accel=list(a), family=p.family, primitive=p.name))
                     for p in chain for e, a in zip(p.edges, p.accelerations))

    def apply(self, decisions):
        event = self.next_event()
        if event['kind'] == 'done':
            raise RuntimeError('episode already finished')
        work = self.clone()
        work._apply(event, decisions)
        entry = dict(slot=self.state.n, kind=event['kind'], choices=deepcopy(decisions))
        if event['kind'] == 'commitments':
            entry['requests'] = event['requests']
        work.decisions.append(freeze(entry))
        work._cached_event = None
        self.__dict__.update(work.__dict__)

    def _apply(self, event, decisions):
        kind = event['kind']; U = self.state.p['U']
        if not isinstance(decisions, dict):
            raise ValueError('decisions must be a mapping')
        if kind == 'layers':
            if set(decisions) != {'layers'}:
                raise ValueError('layer phase requires only layers')
            layers = decisions['layers']
            if len(layers) != U or any(isinstance(e, bool) or not isinstance(e, (int, np.integer)) or
                    not 0 <= e < len(self.lib.heights) or not event['masks'][u][e] for u, e in enumerate(layers)) or len(set(layers)) != U:
                raise ValueError('layers must be distinct legal integer indices')
            self.layers = list(layers)
            for u, ell in enumerate(layers):
                p = self.takeoffs[u, ell]
                chain = (p,)+(self.lib.hover(self.initial_nodes[u], self.lib.heights[ell]),)*(self.takeoff_end-p.duration)
                self.schedules[u] = self._rows(chain)
        elif kind == 'commitments':
            if set(decisions) != {'commitments'}:
                raise ValueError('commitment phase requires only commitments')
            choices = decisions['commitments']
            required = {r['uav'] for r in event['requests']}
            if set(choices) != required:
                raise ValueError('provide exactly the requested UAV commitments')
            for req in event['requests']:
                u = req['uav']; choice = choices[u]
                if set(choice) != {'T', 'route'}:
                    raise ValueError('each commitment requires exactly T and route')
                T, route = choice['T'], choice['route']
                if isinstance(T, bool) or not isinstance(T, (int, np.integer)):
                    raise ValueError('integer target slot required')
                if route == 'keep':
                    if not req['keep'] or T != req['keep_T']:
                        raise ValueError('keep requires the unchanged committed target')
                    if T == self.state.n:
                        self.approved_starts.add((u, req['pointer'], T))
                else:
                    if isinstance(route, bool) or not isinstance(route, (int, np.integer)) or not 0 <= route < len(req['routes']):
                        raise ValueError('invalid route candidate')
                    r = req['routes'][route]
                    if not r['legal'] or not r['earliest'] <= T <= r['latest']:
                        raise ValueError('target slot/route is outside the exact reachability mask')
                    node, start = self._canonical(u); h = self.lib.heights[self.layers[u]]
                    target = Target('endpoint', uav=u) if req['endpoint'] else Target('service', task=req['visit'][0], phase=req['visit'][1])
                    plan = self.planner.plan(self.lib.point(node, h), np.zeros(3), start, target, T, h, route_path=r['path'])
                    if plan.status != 'ok':
                        raise Uncertified(f'accepted mask did not yield a plan: {plan.status}')
                    chain = plan.chain
                    if not req['endpoint']:
                        service = min(self.lib.service(*req['visit'], h), key=lambda p: (p.E, p.name))
                        chain += (service,)
                    self.schedules[u] = self.schedules[u][:start]+self._rows(chain)
                    self.commitments[u] = dict(T=int(T), route=int(route), path=r['path'], pointer=req['pointer'],
                                               endpoint=req['endpoint'], end=start+sum(p.duration for p in chain))
                    if T == self.state.n:
                        self.approved_starts.add((u, req['pointer'], T))
                self.pending.discard(u); self.deferrals.discard(u)
        elif kind == 'locations':
            if set(decisions) != {'locations'} or set(decisions['locations']) != set(event['tasks']):
                raise ValueError('provide exactly the release-task locations')
            for k, m in decisions['locations'].items():
                if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m not in event['legal_locations'][k]:
                    raise ValueError('illegal execution location')
                self.locations[k] = int(m)
        elif kind == 'priorities':
            if set(decisions) != {'priorities'}:
                raise ValueError('priority phase requires only priorities')
            n = self.state.n
            if any(len(rows) <= n for rows in self.schedules):
                raise Uncertified('no committed action for the current slot')
            accel = [self.schedules[u][n]['accel'] for u in range(U)]
            tic = perf_counter(); allocation = allocate(self.state, accel, self.locations, decisions['priorities'],
                                                       service=self.service)
            self.timings['allocator_s'].append(perf_counter()-tic)
            starts = [bool(c and not c['endpoint'] and c['T'] == n) for c in self.commitments]
            actions = SlotActions([self.schedules[u][n]['edge'] for u in range(U)], accel, starts,
                                  self.locations, allocation.sigma, allocation.power, allocation.f)
            pointers = self.state.visit_pointer.copy()
            tic = perf_counter()
            slot = step_v2(self.state, actions, on_infeasible='record')
            if self.service is not None:
                self.service.advance(self.state, slot)
            self.timings['step_s'].append(perf_counter()-tic)
            for u in range(U):
                if pointers[u] != self.state.visit_pointer[u]:
                    self.pending.add(u)

    def result(self):
        if self.state.n != self.state.N:
            raise RuntimeError('result requires a completed episode')
        logs = self.state.logs()
        if self.terminal_replay:
            audit = self.state.physics.replay(thaw(self.state.instance), logs, 'audit')
        else:
            if self.state.layer != 'audit' or self.service is None:
                raise ValueError('training without terminal replay requires audit execution and certified service')
            tic = perf_counter()
            audit = self.state.result()
            audit.pop('C')
            audit['wall_time_s'] = perf_counter()-tic
        result = dict(logs=logs, audit=audit, C=sum(t['completed'] for t in audit['tasks']),
                    certified_service=self.service is not None,
                    service_events=[{name: event.as_dict() for name, event in st.events.items()}
                                    for st in self.task_states()],
                    energies_J=[u['E_u'] for u in audit['uavs']], conditions=audit['conditions'],
                    decisions=[thaw(d) for d in self.decisions], timings=deepcopy(self.timings),
                    restrictions=dict(static_layers=True, battery_mask=False, K_routes=4,
                                      synchronized_takeoff_end=self.takeoff_end))
        if self.state.layer == 'audit':
            result.update(execution_layer='audit', terminal_replay=self.terminal_replay,
                          audit_source='independent_replay' if self.terminal_replay else 'executor_audit_state')
        return result
