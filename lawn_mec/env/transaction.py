from time import perf_counter
import numpy as np
from .geometry import Box, trajectory_box_residual
from .queue import inject, serve, advance_visit
from .projection import project
from .checker import check_slot, slot_integrals, service_speed
from .model import motion


def boundary(env, actions):
    for u in range(env.p['U']):
        if env.edge[u] is not None:
            edge = env.instance['edges'][env.edge[u]]
            if env.n > env.edge_start[u] and Box.from_dict(env.instance['nodes'][edge['target']]).contains(env.q[u], env.cfg['epsilon_num']):
                env.node[u] = edge['target']; env.at_node[u] = True
                env.route[u][-1]['arrival'] = {'available': True, 'value': env.n}
    for k, st in enumerate(env.states):
        if st.events['g'].value == env.n and st.location is None:
            u = env.tasks[k]['u']; chosen = actions[u].get('location', 0)
            if isinstance(chosen, dict): chosen = chosen.get(k, chosen.get(str(k), 0))
            if not isinstance(chosen, (int, np.integer)) or chosen not in range(env.p['M']+1):
                raise ValueError('location must be an integer within the location mask')
            st.location = int(chosen)
        if st.location is not None: inject(env.tasks[k], st, env.n)


def next_visit(env, u):
    pointer = env.visit_pointer[u]
    return env.instance['visits'][u][pointer] if pointer < len(env.instance['visits'][u]) else None


def visit_allowed(env, u, accel):
    visit = next_visit(env, u)
    if visit is None: return False
    k, phase = visit; t, st = env.tasks[k], env.states[k]
    if st.J[phase] != 0 or env.N-env.n < t['L_'+phase]: return False
    if phase == 'a' and (not st.events['r'].available or st.events['r'].value > env.n): return False
    region = Box.from_dict(t['S_'+phase])
    return bool(trajectory_box_residual(env.q[u], env.v[u], accel, region, env.p['delta']) <= env.cfg['epsilon_num']
                and service_speed(env.v[u], accel, t['V_'+phase], env.p['delta']) <= env.cfg['epsilon_num'])


def request_options(env, u, prospective_location=None):
    options = [None]
    for k, st in enumerate(env.states):
        if env.tasks[k]['u'] != u: continue
        location = st.location
        uplink, downlink = st.W[0], st.W[2]
        if st.events['g'].value == env.n and location is None and prospective_location is not None:
            location = prospective_location
            if location: uplink = env.tasks[k]['D']
        if location is not None and location > 0:
            options.extend((k, direction) for work, direction in ((uplink, 'ul'), (downlink, 'dl')) if work > 0)
    return options


def step(env, actions):
    if env.n >= env.N: raise RuntimeError('episode already finished')
    if len(actions) != env.p['U']: raise ValueError('one action per UAV required')
    for action in actions:
        for key in ('priority', 'power'):
            value = action.get(key, 0.)
            if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError(f'{key} must lie in [0,1]')
        if np.asarray(action.get('accel', [0, 0, 0])).shape != (3,) or not np.all(np.isfinite(action.get('accel', [0, 0, 0]))):
            raise ValueError('accel must be a finite three-vector')
    started = perf_counter(); p = env.p; dt = p['delta']; K = len(env.tasks)
    boundary(env, actions)
    diagnostics = {'viol_projection_empty': 0, 'viol_endpoint_unreachable': 0}
    accelerations, active = [], []
    for u, action in enumerate(actions):
        if env.at_node[u]:
            options = [i for i, e in enumerate(env.instance['edges']) if e['source'] == env.node[u]]
            chosen = action.get('edge_choice')
            if chosen not in options:
                chosen = next(i for i in options if env.instance['edges'][i]['target'] == env.node[u])
            env.edge[u] = chosen; env.edge_start[u] = env.n; env.at_node[u] = False
            env.route[u].append({'edge': chosen, 'start': env.n, 'arrival': {'available': False, 'value': None}})
        box = Box.from_dict(env.instance['edges'][env.edge[u]]['box']).erode(p['clearance'])
        visit = next_visit(env, u); service_box = service_limit = None
        continuing = visit is not None and 0 < env.states[visit[0]].J[visit[1]] < env.tasks[visit[0]]['L_'+visit[1]]
        if continuing:
            k, phase = visit
            node_box = Box.from_dict(env.instance['nodes'][env.tasks[k]['z_'+phase]]).erode(p['clearance'])
            service_box = Box.from_dict(env.tasks[k]['S_'+phase]).intersect(node_box)
            service_limit = env.tasks[k]['V_'+phase]
        candidate = np.asarray(action.get('accel', [0, 0, 0]), float)
        if env.n == env.N-1: candidate = 2*(np.array(env.instance['qF'][u])-env.q[u]-dt*env.v[u])/dt**2
        continuous_guard = getattr(env, 'motion_guard', None)
        result = (continuous_guard(env, u, action, env.edge[u], box, service_box, service_limit)
                  if continuous_guard is not None else
                  project(candidate, env.q[u], env.v[u], box, p, env.cfg, service_box, service_limit))
        diagnostics['viol_projection_empty'] += int(not result['success'])
        diagnostics['viol_endpoint_unreachable'] += int(env.n == env.N-1 and not result['candidate_in_set'])
        a = result['accel']; accelerations.append(a)
        if continuing or (action.get('visit_start', False) and visit_allowed(env, u, a)):
            active.append(tuple(visit))
    sigma = np.zeros((K, 2), int); power = np.zeros(K); requests = {}
    for u, action in enumerate(actions):
        req = action.get('request'); req = tuple(req) if req is not None else None
        if req is None or req not in request_options(env, u): continue
        k, direction = req; m = env.states[k].location
        requests.setdefault(m, []).append((-float(action.get('priority', 0)), k, direction, u))
    for m, contenders in requests.items():
        _, k, direction, u = min(contenders)
        sigma[k, 0 if direction == 'ul' else 1] = 1
        power[k] = actions[u].get('power', 0.)*p['P_ul']*sigma[k, 0]
    f = np.zeros(K)
    for u, action in enumerate(actions):
        local = [k for k, st in enumerate(env.states) if env.tasks[k]['u'] == u and st.location == 0 and st.W[1] > 0]
        shares = action.get('cpu_share', [])
        if isinstance(shares, dict): weights = np.array([shares.get(k, shares.get(str(k), 0.)) for k in local], float)
        else:
            shares = np.asarray(shares, float)
            if shares.size not in (0, len(local), len(local)+1, K, K+1): raise ValueError('cpu_share length does not match active/local or full task list')
            weights = shares[local] if shares.size in (K, K+1) else shares[:len(local)] if shares.size else np.zeros(len(local))
        if np.any(~np.isfinite(weights)) or np.any(weights < 0): raise ValueError('CPU weights must be finite and nonnegative')
        total = sum(shares.values()) if isinstance(shares, dict) else np.sum(shares)
        weights = weights/max(1., total)
        for k, share in zip(local, weights): f[k] = min(share*p['F_local'], env.states[k].W[1]/dt)
    for m in range(1, p['M']+1):
        ks = [k for k, st in enumerate(env.states) if st.location == m and st.W[1] > 0]
        def window(k):
            t, st = env.tasks[k], env.states[k]
            return (min(t['d']-t['L_a']*dt, st.events['s'].value*dt+t['H']), k)
        remaining = p['G_total']-env.instance['load'][m-1][env.n]
        for k in sorted(ks, key=window): f[k] = min(env.states[k].W[1]/dt, remaining); remaining -= f[k]
    slot = {'n': env.n, 'q': env.q.tolist(), 'v': env.v.tolist(), 'edge': list(env.edge), 'accel': np.array(accelerations).tolist(),
            'active': [list(x) for x in active], 'locations': [st.location for st in env.states],
            'W': [st.W.tolist() for st in env.states], 'sigma': sigma.tolist(), 'power': power.tolist(), 'f': f.tolist(),
            'diagnostics': diagnostics}
    tic = perf_counter(); guard = check_slot(env.instance, slot, 'guard', env.cfg)
    slot['guard_wall_time_s'] = perf_counter()-tic; slot['guard'] = guard
    mu, parts = slot_integrals(env.instance, slot, 'guard', env.cfg)
    slot['mu'], slot['energy_parts'] = np.asarray(mu).tolist(), np.asarray(parts).tolist()
    for k, phase in active:
        advance_visit(env.states[k], phase, env.tasks[k]['L_'+phase], env.n)
        if env.states[k].J[phase] == env.tasks[k]['L_'+phase]: env.visit_pointer[env.tasks[k]['u']] += 1
    for k, st in enumerate(env.states): serve(st, mu[k], env.n)
    env.q, env.v = motion(env.q, env.v, np.array(accelerations), dt)
    env.B -= np.array(parts).sum(axis=1); env.n += 1
    if env.n == env.N:
        boundary(env, [{'location': action.get('terminal_location', 0)} for action in actions])
        slot['terminal_locations'] = [st.location for st in env.states]
    slot['wall_time_s'] = perf_counter()-started; env.logs.append(slot)
    return env.observation(), -float(np.sum(parts)), env.n == env.N, {'guard': guard, **diagnostics}
