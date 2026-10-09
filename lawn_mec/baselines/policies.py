import copy
import math
from time import perf_counter
import numpy as np
from lawn_mec.env.env import Episode
from lawn_mec.env.geometry import Box, segment_box
from lawn_mec.env.model import gain, motion
from lawn_mec.env.queue import arrivals
from .common import (config, centers, node_of, choose_path, profile_samples,
                     mean_gains, line_profile, window_upper, just_enough_cpu,
                     layer_heights, route_segments, dijkstra)


def edf_request(env, candidates):
    return min((req for req in candidates if req is not None),
               key=lambda req: (window_upper(env.tasks[req[0]], env.states[req[0]],
                                              env.p['delta']), req[0], req[1]),
               default=None)


class Navigator:
    def __init__(self, instance, algo, cfg, path_overrides=None, speed_overrides=None, segment_subdivisions=1, plan_only=False, plan_cache=None):
        self.instance, self.cfg = instance, cfg
        self.legs, self.diagnostics, self.samples = [], [], []
        self.leg = [0]*instance['params']['U']; self.segment = [0]*instance['params']['U']
        self.accels = [[] for _ in self.leg]; self.cursor = [0]*len(self.leg)
        self.heights = layer_heights(instance)
        self.paths = []; self.takeoff = []; self.landing = []
        p=instance['params']
        cache = {} if plan_cache is None else plan_cache
        for u, visits in enumerate(instance['visits']):
            height=self.heights[u]; initial=np.asarray(instance['qI'][u],float)
            position=initial.copy(); position[2]=height
            key = ('takeoff', u, tuple(initial), tuple(position))
            if key not in cache: cache[key] = line_profile(initial,position,p,cfg=cfg)
            self.takeoff.append(cache[key])
            final=np.asarray(instance['qF'][u],float); high_final=final.copy(); high_final[2]=height
            key = ('landing', u, tuple(high_final), tuple(final))
            if key not in cache: cache[key] = line_profile(high_final,final,p,cfg=cfg)
            self.landing.append(cache[key])
            source=node_of(instance,initial); legs=[]; paths=[]; samples=[]
            targets=[(k,phase,instance['tasks'][k]['z_'+phase]) for k,phase in visits]
            targets += [(None,None,node_of(instance,final))]
            for j,(k,phase,target) in enumerate(targets):
                if plan_only:
                    key = ('path', source, target)
                    if key not in cache: cache[key] = dijkstra(instance,source,target)
                    path=cache[key]; diag={}
                else:
                    path,diag=choose_path(instance,source,target,algo,cfg)
                if path_overrides is not None: path=path_overrides[u][j]
                finish=(Box.from_dict(instance['tasks'][k]['S_'+phase]).center if k is not None else high_final.copy())
                finish[2]=height
                scales=None if speed_overrides is None else speed_overrides[u][j]
                service=instance['tasks'][k]['S_'+phase] if k is not None else None
                service_limit=instance['tasks'][k]['V_'+phase] if k is not None else None
                key = ('segments', tuple(path), tuple(position), tuple(finish), float(height),
                       None if service is None else (tuple(service['lo']), tuple(service['hi'])), service_limit)
                if not plan_only or key not in cache:
                    segments=route_segments(instance,path,position,finish,height,cfg,scales,service,service_limit)
                    if plan_only: cache[key]=segments
                else: segments=cache[key]
                self.diagnostics.append({'u':u,'leg':j,**diag,'executed_plan':path,'height_m':float(height)})
                for seg in ([] if plan_only else segments):
                    q,v=seg['start'].copy(),seg['initial_velocity'].copy()
                    for acc in seg['accels']:
                        samples.append(motion(q,v,acc,p['delta']/2)[0]);q,v=motion(q,v,acc,p['delta'])
                legs.append(segments);paths.append(path);source,position=target,finish
            self.legs.append(legs);self.paths.append(paths);self.samples.append(samples or [position])
        self.takeoff_slots=max(map(len,self.takeoff),default=0)
        self.landing_start=None
        self.endpoint_deadline = round(p['T']/p['delta'])-1-cfg['n_reserve']
        self.horizontal_deadline = self.endpoint_deadline-1-max(map(len,self.landing),default=0)
        self.return_cache = {}
        self.return_events = []
        self.connected_wait = [None for _ in self.leg]
        self.wait_events = []

    def wait_point(self, u, position):
        p = self.instance['params']
        box = Box.from_dict(self.instance['nodes'][node_of(self.instance, position)]).erode(p['clearance'])
        spacing = p['D_h']+2.
        side = max(2, math.ceil(math.sqrt(p['U'])))
        pitch = spacing*math.sqrt(2.)
        offset = np.array([u % side-(side-1)/2, u//side-(side-1)/2])*pitch
        point = box.center.copy(); point[:2] += offset
        point[2] = p['h_max']-p['clearance']-1.
        if not box.contains(point):
            raise ValueError('connected waiting stations do not fit the eroded node')
        return point

    def wait_action(self, env, u):
        waiting = self.connected_wait[u]
        j = self.leg[u]
        if waiting is None:
            if j >= len(self.instance['visits'][u]) or not self.ready(u): return None
            k, phase = self.instance['visits'][u][j]
            if (env.visit_pointer[u] != j or phase != 'a'
                    or env.states[k].events['r'].value is not None
                    or env.masks(u, np.zeros(3))['visit_start']): return None
            pending = [kk for kk, (t, st) in enumerate(zip(env.tasks, env.states))
                       if t['u'] == u and st.location not in (None, 0)
                       and (st.W[0] > 0 or st.W[2] > 0
                            or (st.events['c'].value == env.n and st.events['r'].value is None))]
            if not pending: return None
            home = env.q[u].copy(); top = self.wait_point(u, home)
            side = top.copy(); side[2] = home[2]
            climb = line_profile(home, side, env.p, cfg=self.cfg)+line_profile(side, top, env.p, cfg=self.cfg)
            event = {'u': u, 'k': k, 'start_slot': env.n, 'height_m': float(top[2]),
                     'position': top.tolist(), 'stalled': False, 'stalled_slot': None,
                     'end_slot': None}
            self.wait_events.append(event)
            waiting = {'home': home, 'side': side, 'accels': climb, 'cursor': 0,
                       'phase': 'up', 'event': event,
                       'last_X': {kk: float(env.states[kk].W[0]) for kk in pending},
                       'last_progress': env.n}
            self.connected_wait[u] = waiting
        current = {kk: float(env.states[kk].W[0]) for kk in waiting['last_X']}
        if any(current[kk] < old-1e-8 for kk, old in waiting['last_X'].items()):
            waiting['last_progress'] = env.n
        waiting['last_X'] = current
        if (any(x > 0 for x in current.values()) and env.n-waiting['last_progress'] >= 20
                and not waiting['event']['stalled']):
            waiting['event'].update(stalled=True, stalled_slot=env.n)
        if waiting['cursor'] < len(waiting['accels']):
            accel = waiting['accels'][waiting['cursor']]; waiting['cursor'] += 1
            return accel
        if waiting['phase'] == 'down':
            waiting['event']['end_slot'] = env.n
            self.connected_wait[u] = None
            return None
        down = (line_profile(env.q[u], waiting['side'], env.p, cfg=self.cfg)
                +line_profile(waiting['side'], waiting['home'], env.p, cfg=self.cfg))
        result_ready = env.states[waiting['event']['k']].events['r'].value is not None
        reserve = env.n+len(down)+self.return_plan(u, waiting['home'])[2] >= self.horizontal_deadline
        if result_ready or reserve:
            waiting['event']['exit_reason'] = 'result_ready' if result_ready else 'return_reserve'
            waiting.update(phase='down', cursor=0,
                           accels=down)
            return self.wait_action(env, u)
        return np.zeros(3)

    def followup_forecast(self, env, k):
        u = env.tasks[k]['u']; samples = []
        for j in range(env.visit_pointer[u], len(self.instance['visits'][u])):
            segments = self.legs[u][j]
            for seg in segments:
                q, v = seg['start'].copy(), seg['initial_velocity'].copy()
                for accel in seg['accels']:
                    samples.append(motion(q, v, accel, env.p['delta']/2)[0])
                    q, v = motion(q, v, accel, env.p['delta'])
            kk, phase = self.instance['visits'][u][j]
            if (kk, phase) == (k, 'a'): break
            point = segments[-1]['finish'] if segments else (samples[-1] if samples else env.q[u])
            samples.extend([point.copy() for _ in range(env.tasks[kk]['L_'+phase])])
        return samples, env.n+len(samples)

    def return_plan(self, u, position):
        key = (u, tuple(np.round(position, 7)))
        if key not in self.return_cache:
            final = np.asarray(self.instance['qF'][u], float).copy()
            final[2] = self.heights[u]
            path = dijkstra(self.instance, node_of(self.instance, position), node_of(self.instance, final))
            segments = route_segments(self.instance, path, position, final, self.heights[u], self.cfg)
            self.return_cache[key] = (path, segments, sum(len(s['accels']) for s in segments))
        return self.return_cache[key]

    def reserve_return(self, env, u):
        j = self.leg[u]
        if j == len(self.legs[u])-1 or np.linalg.norm(env.v[u]) > self.cfg['position_tolerance']:
            return
        segments = self.legs[u][j]
        at_start = self.segment[u] == 0 and self.cursor[u] == 0
        if not at_start and not self.ready(u): return
        k, phase = self.instance['visits'][u][j]
        task = env.tasks[k]
        progress = env.states[k].J[phase]
        if 0 < progress < task['L_'+phase]: return
        if env.visit_pointer[u] > j: return
        end = segments[-1]['finish'] if at_start and segments else env.q[u]
        remaining = sum(len(s['accels']) for s in segments) if at_start else 0
        return_slots = self.return_plan(u, end)[2]
        wait_slots = 0
        if (phase == 'a' and env.states[k].events['r'].value is None
                and any(t['u'] == u and st.location not in (None, 0) and (st.W[0] > 0 or st.W[2] > 0)
                        for t, st in zip(env.tasks, env.states))):
            top = self.wait_point(u, end); side = top.copy(); side[2] = end[2]
            wait_slots = sum(len(line_profile(a, b, env.p, cfg=self.cfg))
                             for a, b in ((end, side), (side, top), (top, side), (side, end)))
        if remaining == return_slots == 0 and not self.landing[u]: return
        if env.n+remaining+wait_slots+task['L_'+phase]+return_slots <= self.horizontal_deadline:
            return
        path, returning, duration = self.return_plan(u, env.q[u])
        self.return_events.append({'u': u, 'n': env.n, 'next_visit': j,
                                   'horizontal_end_slot': env.n+duration,
                                   'endpoint_deadline_slot': self.endpoint_deadline,
                                   'path': path})
        self.legs[u][-1] = returning
        self.paths[u][-1] = path
        self.leg[u] = len(self.legs[u])-1
        self.segment[u] = self.cursor[u] = 0

    def ready(self,u):
        return self.segment[u] >= len(self.legs[u][self.leg[u]])

    def followup_travel_slots(self, env, k):
        u = env.tasks[k]['u']; j = self.leg[u]
        target = next(i for i, visit in enumerate(self.instance['visits'][u])
                      if tuple(visit) == (k, 'a'))
        if j > target:
            return None
        travel = 0
        for leg in range(j, target+1):
            for index, segment in enumerate(self.legs[u][leg]):
                if leg == j and index < self.segment[u]: continue
                used = max(0, self.cursor[u]-1) if leg == j and index == self.segment[u] else 0
                travel += len(segment['accels'])-used
        return travel

    def action(self, env, u):
        options=env.masks(u)['edge_choice']
        def edge_for(path):
            if not options: return env.edge[u]
            source=env.instance['edges'][options[0]]['source']
            if source in path and path.index(source)<len(path)-1:
                target=path[path.index(source)+1]
                return next(i for i in options if env.instance['edges'][i]['target']==target)
            return self._edge(env,u,None,options)
        if env.n<self.takeoff_slots:
            aa=self.takeoff[u]
            return self._edge(env,u,None,options), aa[env.n] if env.n<len(aa) else np.zeros(3)
        if self.connected_wait[u] is not None:
            accel = self.wait_action(env, u)
            if accel is not None: return self._edge(env,u,None,options), accel
        while True:
            self.reserve_return(env, u)
            j=self.leg[u];segments=self.legs[u][j];idx=self.segment[u]
            if idx<len(segments):
                seg=segments[idx]
                if self.cursor[u]<len(seg['accels']):
                    a=seg['accels'][self.cursor[u]];self.cursor[u]+=1
                    return edge_for(self.paths[u][j]),a
                self.segment[u]+=1;self.cursor[u]=0;continue
            if j<len(self.legs[u])-1 and env.visit_pointer[u]>j:
                self.leg[u]+=1;self.segment[u]=0;continue
            accel = self.wait_action(env, u)
            if accel is not None: return self._edge(env,u,None,options), accel
            terminal=all(self.leg[w]==len(self.legs[w])-1 and self.ready(w) for w in range(env.p['U']))
            if terminal and self.landing_start is None: self.landing_start=env.n+1
            if self.landing_start is not None and env.n>=self.landing_start:
                n=env.n-self.landing_start
                aa=self.landing[u]
                return self._edge(env,u,None,options), aa[n] if n<len(aa) else np.zeros(3)
            return self._edge(env,u,None,options),np.zeros(3)

    @staticmethod
    def _edge(env, u, wanted, options):
        if not options: return env.edge[u]
        if wanted in options: return wanted
        return next(i for i in options if env.instance['edges'][i]['source'] == env.instance['edges'][i]['target'])


class BaselinePolicy:
    def __init__(self, instance, algo='B1', cfg=None, *, locations=None, paths=None, speeds=None,
                 visit_delays=None, visit_slots=None, request_rule=None, b1_logs=None, segment_subdivisions=1):
        if algo not in ('B1', 'B2', 'B3'): raise ValueError('policy must be B1, B2 or B3')
        self.instance, self.algo, self.cfg = instance, algo, cfg or config()
        self.navigator = Navigator(instance, 'B3' if algo == 'B3' else 'B1', self.cfg, paths, speeds, segment_subdivisions)
        self.fixed_locations = locations
        self.visit_delays, self.visit_slots = visit_delays or {}, visit_slots or {}
        self.first_allowed = {}; self.request_arrival = {}; self.request_rule = request_rule or 'edf'
        self.locations = {}; self.location_estimates = {}; self.power_unreachable = 0
        self.bs_assignments = {}
        self.power_decisions = []; self.unreachable_tasks = {}
        if algo == 'B2':
            if b1_logs is None: b1_logs = rollout(instance, BaselinePolicy(instance, 'B1', self.cfg))[0].logs
            samples = [[motion(np.array(s['q'][u]), np.array(s['v'][u]), np.array(s['accel'][u]), instance['params']['delta']/2)[0] for s in b1_logs] for u in range(instance['params']['U'])]
        else: samples = self.navigator.samples
        self.gains = np.array([mean_gains(instance, points) for points in samples])

    def upload_slots(self, task, m):
        p = self.instance['params']; g = self.gains[task['u'], m-1]
        rate = p['bandwidth']*np.log2(1+p['P_ul']*g/(p['N0']*p['bandwidth']))
        return max(1, math.ceil(task['D']/(p['delta']*rate)))

    def estimate_location(self, env, k):
        t, st, p = env.tasks[k], env.states[k], env.p; dt = p['delta']; u = t['u']
        upper = min(window_upper(t, st, dt), p['T']-t['L_a']*dt); options = []
        samples, earliest_visit = self.navigator.followup_forecast(env, k)
        local_slots = max(1, math.ceil(t['C']/(dt*p['F_local'])))
        full, residual = divmod(t['C'], dt*p['F_local'])
        local_energy = full*dt*p['kappa']*p['F_local']**3+dt*p['kappa']*(residual/dt)**3
        options.append({'location': 0, 'ready_slot': env.n+local_slots, 'energy_J': local_energy})
        for m in range(1, p['M']+1):
            assigned = [row for kk, row in self.bs_assignments.items()
                        if kk != k and row['location'] == m]
            upload_sum = sum(row['upload_slots'] for row in assigned)
            queue = len(assigned)+upload_sum
            bs = env.instance['bs'][m-1]
            visible = [not any(segment_box(q, bs, Box.from_dict(b)) for b in env.instance['buildings'])
                       for q in samples]
            gains = [gain(q, bs, env.instance['buildings'], p) if los else 0.
                     for q, los in zip(samples, visible)]
            cursor = queue; ul = comp = dl = 0
            def radio(work, power):
                nonlocal cursor
                served = 0
                while work > 1e-8 and cursor < len(samples):
                    g = gains[cursor]; cursor += 1
                    if g <= 0: continue
                    work -= dt*p['bandwidth']*np.log2(1+power*g/(p['N0']*p['bandwidth']))
                    served += 1
                return served, work <= 1e-8
            ul, uploaded = radio(t['D'], p['P_ul'])
            work = t['C']
            if uploaded:
                while work > 1e-8 and cursor < len(samples) and env.n+cursor < env.N:
                    work -= dt*(p['G_total']-env.instance['load'][m-1][env.n+cursor])
                    cursor += 1; comp += 1
            if uploaded and work <= 1e-8:
                dl, downloaded = radio(t['O'], p['P_dl'])
            else: downloaded = False
            ready = env.n+cursor if downloaded else env.N+1
            options.append({'location': m, 'ready_slot': ready,
                            'assigned_tasks': len(assigned), 'assigned_upload_slots': upload_sum,
                            'queue_slots': queue, 'upload_slots': ul,
                            'planned_followup_slot': earliest_visit,
                            'los_slots': sum(visible), 'forecast_slots': len(samples),
                            'eligible': bool(downloaded and ready <= earliest_visit-2),
                            'compute_slots': comp, 'downlink_slots': dl,
                            'power_W': p['P_ul'],
                            'energy_J': ul*dt*(p['P_ul']/p['eta_tx']+p['P_tx'])+dl*dt*p['P_rx']})
        for option in options:
            option['followup_slot'] = max(earliest_visit, option['ready_slot'])
            option['completion_slot'] = option['followup_slot']+t['L_a']
            option['lateness_s'] = max(0., option['followup_slot']*dt-upper)
        selected = min((x for x in options if x['location'] == 0 or x['eligible']),
                       key=lambda x: (x['completion_slot'], x['energy_J'], x['location']))
        self.location_estimates[k] = options
        self.bs_assignments[k] = {'location': selected['location'],
                                  'upload_slots': selected.get('upload_slots', 0)}
        return selected['location']

    def _resources(self, env):
        states = copy.deepcopy(env.states)
        for k, (t, st) in enumerate(zip(env.tasks, states)):
            if st.events['g'].value == env.n and st.location is None:
                if self.fixed_locations is not None: m = self.fixed_locations[k]
                elif self.algo == 'B1': m = 0
                elif self.algo == 'B2': m = int(np.argmax(self.gains[t['u']]))+1
                else: m = self.estimate_location(env, k)
                self.locations[k] = m; st.location = m
            if st.location is not None: st.W += arrivals(t, st, env.n)
        return states

    def power(self, env, k, state, accel):
        return 1.

    def adaptive_power(self, env, k, state, accel):
        p, t, dt = env.p, env.tasks[k], env.p['delta']
        m, u = state.location, t['u']
        upper = math.floor(window_upper(t, env.states[k], dt)/dt)
        xs, ws = np.polynomial.legendre.leggauss(env.cfg['guard_nodes'])
        gains = np.array([gain(motion(env.q[u], env.v[u], accel, (x+1)*dt/2)[0],
                               env.instance['bs'][m-1], env.instance['buildings'], p) for x in xs])
        def capacity(power):
            return float(dt*p['bandwidth']*np.dot(ws/2, np.log2(1+power*gains/(p['N0']*p['bandwidth']))))
        G = p['G_total']-env.instance['load'][m-1][env.n]
        comp = math.ceil(t['C']/(dt*G))
        dl = math.ceil(t['O']/capacity(p['P_dl']))
        travel = self.navigator.followup_travel_slots(env, k)
        budget = upper-env.n-comp-dl-(travel or 0)
        cap_work = capacity(p['P_ul'])
        reason = ('followup_removed_by_return_reserve' if travel is None else
                  'nonpositive_upload_budget' if budget <= 0 else
                  'maximum_power_capacity_below_remaining_backlog' if budget*cap_work < state.W[0] else None)
        value = p['P_ul']
        if reason is None:
            lo, hi = 0., p['P_ul']
            for _ in range(self.cfg['power_bisections']):
                mid = (lo+hi)/2
                if budget*capacity(mid) >= state.W[0]: hi = mid
                else: lo = mid
            value = min(p['P_ul'], hi*self.cfg['b2']['power_safety_factor'])
        row = {'k': k, 'u': u, 'n': env.n, 'window_upper_slot': upper,
               'compute_slots': comp, 'downlink_slots': dl, 'travel_slots': travel,
               'n_budget': budget, 'remaining_X': float(state.W[0]),
               'max_slot_capacity_bits': cap_work, 'power_W': value,
               'reachable_under_current_slot_forecast': reason is None, 'reason': reason}
        self.power_decisions.append(row)
        if reason is not None:
            self.power_unreachable += 1
            self.unreachable_tasks.setdefault(k, row)
        return value/p['P_ul']

    def action(self, env):
        states = self._resources(env); workloads = [s.W for s in states]; locations = [s.location for s in states]
        cpu = just_enough_cpu(env, workloads, locations); result = []
        for u in range(env.p['U']):
            edge, accel = self.navigator.action(env, u)
            j = env.visit_pointer[u]; visit = env.instance['visits'][u][j] if j < len(env.instance['visits'][u]) else None
            start = False
            if (visit is not None and self.navigator.leg[u] == j and self.navigator.ready(u)
                    and env.n >= self.navigator.takeoff_slots
                    and np.linalg.norm(env.v[u]) <= self.cfg['position_tolerance']
                    and np.linalg.norm(accel) <= self.cfg['position_tolerance']
                    and (env.n+env.tasks[visit[0]]['L_'+visit[1]] < env.N
                         or np.linalg.norm(env.q[u]-env.instance['qF'][u]) <= self.cfg['position_tolerance'])
                    and env.masks(u, accel)['visit_start']):
                key = (u, j); self.first_allowed.setdefault(key, env.n)
                start = (env.n == self.visit_slots[key] if key in self.visit_slots else env.n >= self.first_allowed[key]+self.visit_delays.get(key, 0))
            candidates = []
            for k, st in enumerate(states):
                if env.tasks[k]['u'] != u or st.location in (0, None): continue
                for idx, direction in ((0, 'ul'), (2, 'dl')):
                    if st.W[idx] > 0:
                        req = (k, direction); self.request_arrival.setdefault(req, env.n)
                        candidates.append(req)
            def rank(req):
                k = req[0]; upper = window_upper(env.tasks[k], env.states[k], env.p['delta'])
                return (self.request_arrival[req], upper, k, req[1])
            req = ((min(candidates, key=rank) if candidates else None)
                   if self.request_rule == 'fcfs' else edf_request(env, candidates))
            priority = 0.; power = 0.
            if req is not None:
                upper = window_upper(env.tasks[req[0]], env.states[req[0]], env.p['delta'])
                priority = float(0.5-np.arctan(upper/env.p['T'])/np.pi)
                if req[1] == 'ul': power = self.power(env, req[0], states[req[0]], accel)
            result.append({'edge_choice': edge, 'accel': np.asarray(accel).tolist(), 'visit_start': bool(start),
                           'location': {k: m for k, m in self.locations.items() if env.tasks[k]['u'] == u},
                           'request': req, 'priority': priority, 'power': power,
                           'cpu_share': {k: w for k, w in cpu.items() if env.tasks[k]['u'] == u}})
            if env.n == env.N-1:
                terminal = {}
                for k, (task, st) in enumerate(zip(env.tasks, env.states)):
                    if task['u'] != u or st.location is not None: continue
                    if self.fixed_locations is not None: m = self.fixed_locations[k]
                    elif self.algo == 'B1': m = 0
                    elif self.algo == 'B2': m = int(np.argmax(self.gains[u]))+1
                    else: m = self.estimate_location(env, k)
                    terminal[k] = m
                result[-1]['terminal_location'] = terminal
        return result

    def diagnostics(self):
        return {'routes': self.navigator.diagnostics, 'mean_gains': self.gains.tolist(),
                'return_reserve': self.navigator.return_events,
                'connected_wait': self.navigator.wait_events,
                'endpoint_deadline_slot': self.navigator.endpoint_deadline,
                'planned_endpoint_slots': [None if self.navigator.landing_start is None else
                                           self.navigator.landing_start+len(aa)
                                           for aa in self.navigator.landing],
                'locations': self.locations, 'location_estimates': self.location_estimates,
                'bs_assignments': self.bs_assignments, 'uplink_power_rule': 'maximum',
                'power_forecast_unreachable': self.power_unreachable, 'request_rule': self.request_rule,
                'power_decisions': self.power_decisions, 'unreachable_tasks': self.unreachable_tasks}


def rollout(instance, policy, seed=0):
    env = Episode(instance, seed); actions = []; inference = []
    while env.n < env.N:
        tic = perf_counter(); action = policy.action(env); inference.append(perf_counter()-tic)
        actions.append(copy.deepcopy(action)); env.step(action)
    return env, actions, inference
