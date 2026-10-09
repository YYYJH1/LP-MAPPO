from time import perf_counter
from fractions import Fraction
from functools import lru_cache
import numpy as np
from .geometry import Box, trajectory_box_residual
from .model import (motion, shaft_power, power_floor, shaft_interval, rate_interval,
                    slot_capacity, flight_energy, cpu_energy, radio_energy, payload_energy, frequency, array_signature, shaft_parameters, _kernel_iv, _interval_kernel_inputs, _iv_separation, _INTERVAL_POWERS)
from .model import _small_polyroots, _small_polysquare
from .interval import Interval, iv, enclose_integral, certify_lower
from .params import checker_config
from lawn_mec.eval.registry import IDS, FIELDS, verdict, merge


def flight_speed(v, a, p):
    return max(max(np.linalg.norm(w[:2])-p['V_h'], w[2]-p['V_up'], -w[2]-p['V_down'])
               for w in (v, v+p['delta']*a))


def service_speed(v, a, limit, delta):
    return max(np.linalg.norm(v), np.linalg.norm(v+delta*a))-limit


def separation_min(q, v, a, p):
    scale = np.array([p['D_h'], p['D_h'], p['D_z']])
    coeff = np.zeros(5)
    for i in range(3):
        c = np.array([q[i], v[i], a[i]/2])/scale[i]
        squared = _small_polysquare(c)
        coeff[:len(squared)] += squared
    derivative = np.arange(1, 5)*coeff[1:]
    roots = _small_polyroots(derivative)
    ts = [0., p['delta']]+[float(z.real) for z in roots if abs(z.imag) < 1e-8 and 0 < z.real < p['delta']]
    return float(min(np.polynomial.polynomial.polyval(t, coeff) for t in ts))


def separation_interval(q, v, a, p, t):
    scale = [p['D_h'], p['D_h'], p['D_z']]
    if _interval_kernel_inputs((q, v, a), t):
        result = _iv_separation(q, v, a, np.asarray(scale, float),
                                (float(t.lo), float(t.hi)), _INTERVAL_POWERS)
        if np.all(np.isfinite(result)): return Interval(*result)
    t = _kernel_iv(t) if isinstance(t, Interval) else t
    result = sum(((_kernel_iv(q[i])+t*v[i]+t**2*a[i]/2)/scale[i])**2 for i in range(3))
    return Interval(result.lo, result.hi)


@lru_cache(maxsize=4096)
def _separation_samples_cached(qkey, vkey, akey, parameters, evaluator):
    arrays = [np.frombuffer(key[2], dtype=key[0]).reshape(key[1])
              for key in (qkey, vkey, akey)]
    p = dict(parameters); dt = p['delta']
    return min(evaluator(*arrays, p, iv(t)).lo for t in (0, dt/2, dt))


def _separation_samples(q, v, a, p):
    return _separation_samples_cached(array_signature(q), array_signature(v),
        array_signature(a), tuple((k, p[k]) for k in ('D_h', 'D_z', 'delta')),
        separation_interval)


@lru_cache(maxsize=4096)
def _power_check_cached(vkey, akey, parameters, settings, layer, functions):
    v = np.frombuffer(vkey[2], dtype=vkey[0]).reshape(vkey[1])
    a = np.frombuffer(akey[2], dtype=akey[0]).reshape(akey[1])
    return _power_check_value(v, a, dict(parameters), dict(settings), layer)


def power_check(v, a, p, cfg, layer):
    if layer == 'audit': return _power_check_value(v, a, p, cfg, layer)
    return dict(_power_check_cached(array_signature(v), array_signature(a),
        shaft_parameters(p, ('delta', 'power_floor_ratio')),
        (('M_eval', cfg['M_eval']),), layer, (shaft_power, power_floor)))


def _power_check_value(v, a, p, cfg, layer):
    floor = power_floor(p)
    if layer == 'audit':
        return certify_lower(lambda t: shaft_interval(v, a, p, t), p['delta'], floor, cfg['interval_budget'])
    ts = np.linspace(0, p['delta'], 3 if layer == 'guard' else cfg['M_eval'])
    vals = shaft_power(v+ts[:, None]*a, a, p)
    margin = 0.
    if layer == 'evaluation':
        h = max(1e-7, p['delta']*1e-5)
        derivatives = abs(shaft_power(v+(ts+h)[:, None]*a, a, p)
                          -shaft_power(v+(ts-h)[:, None]*a, a, p))/(2*h)
        margin = 2*max(derivatives, default=0)*p['delta']/(len(ts)-1)/2
    residual = floor-float(vals.min())
    if layer == 'evaluation':
        return {'status': 'fail' if residual > 0 else 'pass' if residual+margin <= 0 else 'unknown',
                'residual': residual+margin, 'sample_residual': residual, 'derivative_margin': margin}
    return verdict(residual)


def _distance_trajectory_box(q, v, a, box, delta):
    cuts = [0., delta]
    for i in range(3):
        for face in (box.lo[i], box.hi[i]):
            roots = np.polynomial.polynomial.polyroots([q[i]-face, v[i], a[i]/2])
            cuts += [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < delta]
    cuts = sorted(set(cuts)); best = float('inf')
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        midq = motion(q, v, a, (lo+hi)/2)[0]; polynomial = np.zeros(5)
        for i in range(3):
            target = box.lo[i] if midq[i] < box.lo[i] else box.hi[i] if midq[i] > box.hi[i] else None
            if target is not None:
                c = [q[i]-target, v[i], a[i]/2]; sq = np.polynomial.polynomial.polymul(c, c)
                polynomial[:len(sq)] += sq
        roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(polynomial))
        ts = [lo, hi]+[float(r.real) for r in roots if abs(r.imag) < 1e-8 and lo < r.real < hi]
        best = min(best, *(np.polynomial.polynomial.polyval(t, polynomial) for t in ts))
    return np.sqrt(max(0, best))


def corridor_check(q, v, a, edge, instance, cfg, layer):
    p = instance['params']; dt = p['delta']; eps = cfg['epsilon_num']
    box = Box.from_dict(edge['box'])
    inner = trajectory_box_residual(q, v, a, box.erode(p['clearance']), dt)
    original = trajectory_box_residual(q, v, a, box, dt)
    if inner <= eps and instance.get('_geometry_accepted', False):
        return verdict(inner, eps), inner
    if original > eps: return verdict(original, eps), inner
    boxes = [Box.from_dict(e['box']) for e in instance['edges']]
    cuts = [sorted(set([b.lo[i] for b in boxes]+[b.hi[i] for b in boxes])) for i in range(3)]
    obstacles = [Box.from_dict(b) for b in instance['buildings']+instance['no_fly']]
    for x0, x1 in zip(cuts[0][:-1], cuts[0][1:]):
        for y0, y1 in zip(cuts[1][:-1], cuts[1][1:]):
            for z0, z1 in zip(cuts[2][:-1], cuts[2][1:]):
                cell = Box([x0, y0, z0], [x1, y1, z1])
                if not any(b.contains(cell.center) for b in boxes): obstacles.append(cell)
    outer = Box([x[0] for x in cuts], [x[-1] for x in cuts])
    residual = trajectory_box_residual(q, v, a, outer.erode(p['clearance']), dt)
    for obs in obstacles:
        residual = max(residual, p['clearance']-_distance_trajectory_box(q, v, a, obs, dt))
    if layer == 'audit' and residual <= eps:
        return {'status': 'unknown', 'residual': float(residual)}, inner
    return verdict(residual, eps), inner


def check_slot(instance, slot, layer='guard', cfg=None):
    cfg = cfg or checker_config(); p = instance['params']; eps = cfg['epsilon_num']
    dt = p['delta']; tasks = instance['tasks']; n = slot['n']
    q, v, a = (np.array(slot[x], float) for x in ('q', 'v', 'accel'))
    cs = {k: verdict(0) for k in IDS}; inners = []
    fields = {f: verdict(0) for names in FIELDS.values() for f in names}
    for u in range(p['U']):
        edge = instance['edges'][slot['edge'][u]]
        c04, inner = corridor_check(q[u], v[u], a[u], edge, instance, cfg, layer)
        inners.append(inner); cs['C04'] = merge(cs['C04'], c04)
        cs['C11'] = merge(cs['C11'], verdict(flight_speed(v[u], a[u], p), eps))
        cs['C12'] = merge(cs['C12'], verdict(max(np.linalg.norm(a[u, :2])-p['a_h'], abs(a[u, 2])-p['a_z']), eps))
        cs['C13'] = merge(cs['C13'], power_check(v[u], a[u], p, cfg, layer))
    for k, phase in slot['active']:
        u = tasks[k]['u']
        cs['C05'] = merge(cs['C05'], verdict(trajectory_box_residual(q[u], v[u], a[u], Box.from_dict(tasks[k]['S_'+phase]), dt), eps))
        cs['C06'] = merge(cs['C06'], verdict(service_speed(v[u], a[u], tasks[k]['V_'+phase], dt), eps))
    for u in range(p['U']):
        for w in range(u):
            dq, dv, da = q[u]-q[w], v[u]-v[w], a[u]-a[w]
            if layer == 'audit':
                result = certify_lower(lambda t: separation_interval(dq, dv, da, p, t), dt, 1., cfg['interval_budget'])
            elif layer == 'evaluation':
                residual = 1-separation_min(dq, dv, da, p)
                result = {'status': 'unknown' if abs(residual) <= eps else 'fail' if residual > 0 else 'pass',
                          'residual': residual}
            else:
                result = verdict(1-_separation_samples(dq, dv, da, p), eps)
            cs['C14'] = merge(cs['C14'], result)
    sigma, power, f = np.array(slot['sigma']), np.array(slot['power']), np.array(slot['f'])
    locations = slot['locations']; owners = [t['u'] for t in tasks]
    cs['C15'] = verdict(max(0., float(np.max(np.abs(sigma-np.round(sigma)))), float(-sigma.min()), float(sigma.max()-1)))
    for u in range(p['U']):
        indices = [k for k, t in enumerate(tasks) if t['u'] == u]
        fields['viol_tdma_uav'] = merge(fields['viol_tdma_uav'], verdict(float(sigma[indices].sum()-1)))
        fields['viol_cpu_local'] = merge(fields['viol_cpu_local'], verdict(frequency(f, owners, locations, u)-p['F_local'], eps))
        cs['C15'] = merge(cs['C15'], fields['viol_tdma_uav'])
        cs['C17'] = merge(cs['C17'], fields['viol_cpu_local'])
    for m in range(1, p['M']+1):
        indices = [k for k in range(len(tasks)) if locations[k] == m]
        fields['viol_tdma_bs'] = merge(fields['viol_tdma_bs'], verdict(float(sigma[indices].sum()-1)))
        fields['viol_cpu_mec'] = merge(fields['viol_cpu_mec'], verdict(float(f[indices].sum()-(p['G_total']-instance['load'][m-1][n])), eps))
        cs['C15'] = merge(cs['C15'], fields['viol_tdma_bs'])
        cs['C17'] = merge(cs['C17'], fields['viol_cpu_mec'])
    cs['C16'] = verdict(max(float(-power.min()), float((power-p['P_ul']*sigma[:, 0]).max())), eps)
    cs['C17'] = merge(cs['C17'], verdict(float(-f.min()), eps))
    cs['C20'] = verdict(float(any(m is not None and (not isinstance(m, (int, np.integer)) or not 0 <= m <= p['M']) for m in locations)))
    cs['C09'] = verdict(float(any((m in (None, 0) and sigma[k].sum() != 0) or (m is None and f[k] != 0)
                                        for k, m in enumerate(locations))))
    for cid, names in FIELDS.items():
        if cid not in ('C15', 'C17'):
            for name in names: fields[name] = merge(fields[name], cs[cid])
    return {'conditions': cs, 'field_conditions': fields, 'rej_corridor_inner': sum(r > eps for r in inners)}


def slot_integrals(instance, slot, layer, cfg):
    p = instance['params']; tasks = instance['tasks']; dt = p['delta']
    mu = [[iv(0), iv(dt)*x, iv(0)] for x in slot['f']] if layer == 'audit' else np.zeros((len(tasks), 3))
    if layer != 'audit': mu[:, 1] = dt*np.array(slot['f'])
    q, velocities, accel = (np.array(slot[key]) for key in ('q', 'v', 'accel'))
    energy = []
    for k, t in enumerate(tasks):
        m = slot['locations'][k]; u = t['u']
        if m in (0, None): continue
        for j, component in [(0, 0), (1, 2)]:
            sigma = slot['sigma'][k][j]; power = slot['power'][k] if j == 0 else p['P_dl']*sigma
            if layer != 'audit' and (not sigma or power == 0): continue
            args = [q[u], velocities[u], accel[u], instance['bs'][m-1], instance['buildings'], power, sigma, p]
            if layer == 'audit': value = enclose_integral(lambda t: rate_interval(*args, t), dt, cfg['integral_subdivisions'])
            else: value = slot_capacity(*args, layer, cfg)
            mu[k][component] = value
    for u in range(p['U']):
        ks = [k for k, t in enumerate(tasks) if t['u'] == u]
        v, a = velocities[u], accel[u]
        if layer == 'audit':
            ef = dt*p['P0']+enclose_integral(lambda t: shaft_interval(v, a, p, t), dt, cfg['integral_subdivisions'])/p['eta_flight']
        else: ef = flight_energy(v, a, p, layer, cfg)
        F = frequency(slot['f'], [t['u'] for t in tasks], slot['locations'], u)
        ec = cpu_energy(F, p)
        er = radio_energy([slot['power'][k] for k in ks], [slot['sigma'][k][0] for k in ks], [slot['sigma'][k][1] for k in ks], p)
        ep = payload_energy([(k, phase) for k, phase in slot['active'] if k in ks], tasks, p)
        if layer == 'audit':
            F_interval = sum((iv(slot['f'][k]) for k in ks if slot['locations'][k] == 0), iv(0))
            ec = iv(dt)*p['kappa']*F_interval**3
            er = iv(dt)*(sum((iv(slot['power'][k]) for k in ks), iv(0))/p['eta_tx']
                          +iv(p['P_tx'])*sum(slot['sigma'][k][0] for k in ks)
                          +iv(p['P_rx'])*sum(slot['sigma'][k][1] for k in ks))
            ep = iv(dt)*sum((iv(tasks[k]['P_'+phase]) for k, phase in slot['active'] if k in ks), iv(0))
        energy.append([ef, ec, er, ep])
    return mu, energy


def _completion(capacities, release, work, high, numerical=False):
    if release is None: return None
    total = 0. if numerical else Fraction(0)
    for n in range(release, len(capacities)):
        cap = capacities[n]
        value = (cap.hi if high else cap.lo) if isinstance(cap, Interval) else cap
        total += float(value) if numerical else Fraction(value)
        if total >= work: return n+1
    return None


def replay_episode(instance, slots, layer='evaluation', cfg=None):
    from .generator import accept
    from .queue import TaskState, inject, serve, advance_visit
    cfg = cfg or checker_config(); start = perf_counter()
    inst = dict(instance); inst['_geometry_accepted'] = accept(instance)['passed']
    p, tasks = inst['params'], inst['tasks']; N = round(p['T']/p['delta']); dt = p['delta']; eps = cfg['epsilon_num']
    cs = {key: verdict(0) for key in IDS}; states = [TaskState() for _ in tasks]
    positions = np.array(inst['qI']); velocities = np.array(inst['v0']); energies = [[iv(0) for _ in range(4)] for _ in range(p['U'])]
    route_node = [next((i for i, b in enumerate(inst['nodes']) if Box.from_dict(b).contains(q)), None) for q in positions]
    cs['C02'] = verdict(float(any(z is None for z in route_node)))
    current = [None]*p['U']; arrival = [0]*p['U']; arrival_flags = [True]*p['U']; caps = []
    visit_pointer = [0]*p['U']; slot_results = []; diag = {'rej_corridor_inner': 0, 'rej_service_inner': 0}
    cs['C03'] = verdict(float(len(slots) != N))
    fields = {f: verdict(0) for names in FIELDS.values() for f in names}
    if slots:
        cs['C10'] = verdict(float(np.max(np.linalg.norm(np.array(slots[0]['q'])-positions, axis=1))), eps)
        cs['C20'] = verdict(float(max(np.max(np.abs(np.array(slots[0]['v'])-velocities)), np.max(np.abs(slots[0]['W'])))), eps)
    for n, slot in enumerate(slots):
        cs['C09'] = merge(cs['C09'], verdict(float(slot['n'] != n)),
                          verdict(float(np.max(np.abs(positions-np.array(slot['q'])))), eps),
                          verdict(float(np.max(np.abs(velocities-np.array(slot['v'])))), eps))
        for u in range(p['U']):
            chosen = slot['edge'][u]; edge = inst['edges'][chosen]
            if n and current[u] is not None:
                prev = inst['edges'][current[u]]
                arrival_flags[u] = n > arrival[u] and Box.from_dict(inst['nodes'][prev['target']]).contains(positions[u], eps)
                if arrival_flags[u]: route_node[u], arrival[u] = prev['target'], n
            if arrival_flags[u]:
                cs['C01'] = merge(cs['C01'], verdict(float(edge['source'] != route_node[u])))
            else:
                cs['C01'] = merge(cs['C01'], verdict(float(chosen != current[u])))
            current[u] = chosen
        for k, state in enumerate(states):
            m = slot['locations'][k]
            release = state.events['g'].value
            if release == n:
                cs['C20'] = merge(cs['C20'], verdict(float(m is None)))
                state.location = m
            else:
                cs['C20'] = merge(cs['C20'], verdict(float(m != state.location)))
            if state.location is not None: inject(tasks[k], state, n)
            cs['C09'] = merge(cs['C09'], verdict(float(np.max(np.abs(state.W-np.array(slot['W'][k])))), max(eps, tasks[k]['C']*1e-12)))
        active = [tuple(x) for x in slot['active']]
        for u in range(p['U']):
            seq = inst['visits'][u]; idx = visit_pointer[u]
            own_active = [x for x in active if tasks[x[0]]['u'] == u]
            cs['C08'] = merge(cs['C08'], verdict(float(len(own_active) > 1 or any(idx >= len(seq) or list(x) != seq[idx] for x in own_active))))
            if idx < len(seq):
                k, phase = seq[idx]; st = states[k]
                if 0 < st.J[phase] < tasks[k]['L_'+phase]:
                    continuation = verdict(float((k, phase) not in active))
                    fields['viol_visit_continue'] = merge(fields['viol_visit_continue'], continuation)
                    cs['C07'] = merge(cs['C07'], continuation)
        for k, phase in active:
            st = states[k]
            if st.J[phase] >= tasks[k]['L_'+phase]: cs['C07'] = merge(cs['C07'], verdict(1)); continue
            advance_visit(st, phase, tasks[k]['L_'+phase], n)
            if st.J[phase] == tasks[k]['L_'+phase]: visit_pointer[tasks[k]['u']] += 1
        result = check_slot(inst, slot, layer, cfg); slot_results.append(result)
        for key in IDS: cs[key] = merge(cs[key], result['conditions'][key])
        for key, item in result['field_conditions'].items(): fields[key] = merge(fields[key], item)
        diag['rej_corridor_inner'] += result['rej_corridor_inner']
        mu, parts = slot_integrals(inst, slot, layer, cfg)
        caps.append([[iv(x) for x in row] for row in mu])
        guard_mu = np.array(slot['mu']); expected_mu, _ = slot_integrals(inst, slot, 'guard', cfg)
        cs['C09'] = merge(cs['C09'], verdict(float(np.max(np.abs(expected_mu-guard_mu))), max(eps, 1e-10*float(np.max(expected_mu)))))
        for k, st in enumerate(states): serve(st, guard_mu[k], n)
        for u in range(p['U']):
            for j in range(4): energies[u][j] += parts[u][j]
        positions, velocities = motion(positions, velocities, np.array(slot['accel']), dt)
    if len(slots) == N:
        terminal_locations = slots[-1].get('terminal_locations', [st.location for st in states])
        for k, st in enumerate(states):
            if st.events['g'].value == N and st.location is None:
                st.location = terminal_locations[k]
                cs['C20'] = merge(cs['C20'], verdict(float(st.location not in range(p['M']+1))))
            else:
                cs['C20'] = merge(cs['C20'], verdict(float(terminal_locations[k] != st.location)))
            if st.location is not None: inject(tasks[k], st, N)
    residuals = np.linalg.norm(positions-np.array(inst['qF']), axis=1)
    cs['C10'] = merge(cs['C10'], verdict(float(max(residuals)), eps))
    for u, edgeidx in enumerate(current):
        if edgeidx is None or not Box.from_dict(inst['nodes'][inst['edges'][edgeidx]['target']]).contains(positions[u], eps):
            cs['C03'] = merge(cs['C03'], verdict(1))
    task_rows = []
    for k, t in enumerate(tasks):
        st = states[k]; ns, ng, na = (st.events[x].value for x in ('s', 'g', 'a'))
        lower = upper = ng
        stages = (1,) if st.location == 0 else (0, 1, 2)
        for stage in stages:
            work = [t['D'], t['C'], t['O']][stage]
            series = [c[k][stage] for c in caps]
            if stage == 1 and layer == 'audit':
                series = [Fraction(dt)*Fraction(s['f'][k]) for s in slots]
            lower = _completion(series, lower, work, True, layer != 'audit')
            upper = _completion(series, upper, work, False, layer != 'audit')
        if na is None: ready = deadline = fresh = None
        else:
            ready = 'pass' if upper is not None and upper <= na else 'fail' if lower is None or lower > na else 'unknown'
            deadline = 'pass' if (na+t['L_a'])*dt <= t['d'] else 'fail'
            fresh = 'pass' if ns is not None and (na-ns)*dt <= t['H'] else 'fail'
        window = [ready, deadline, fresh]
        cs['C18'] = merge(cs['C18'], *[{'status': x or 'fail', 'residual': 0. if x in ('pass', 'unknown') else 1.} for x in window])
        visits_done = all(st.J[x] == t['L_'+x] for x in ('s', 'a'))
        fields['viol_visit_complete'] = merge(fields['viol_visit_complete'], verdict(float(not visits_done)))
        cs['C07'] = merge(cs['C07'], fields['viol_visit_complete'])
        for name, status in zip(('viol_result_ready', 'viol_deadline', 'viol_freshness'), window):
            fields[name] = merge(fields[name], {'status': status or 'fail', 'residual': 0. if status in ('pass', 'unknown') else 1.})
        events = {x: e.as_dict() for x, e in st.events.items()}
        nr = lower if lower == upper else None
        events['r'] = {'available': nr is not None, 'value': nr}
        task_rows.append({'k': k, 'u': t['u'], 'started': ns is not None, 'completed': visits_done and window == ['pass']*3,
                          'events': events, 'n_s': ns, 'n_g': ng, 'n_r': nr, 'n_a': na,
                          'n_r_interval': [lower, upper], 'location': st.location, 'J_s': st.J['s'], 'J_a': st.J['a'],
                          'result_ready': ready, 'deadline': deadline, 'freshness': fresh,
                          'deadline_slack_s': None if na is None else t['d']-(na+t['L_a'])*dt,
                          'freshness_slack_s': None if na is None or ns is None else t['H']-(na-ns)*dt})
    uav_rows = []
    for u in range(p['U']):
        total = sum(energies[u], iv(0)); B = p['battery']-total
        status = 'pass' if B.lo >= 0 else 'fail' if B.hi < 0 else 'unknown'
        cs['C19'] = merge(cs['C19'], {'status': status, 'residual': -B.lo})
        uav_rows.append({'u': u, 'E_u': (total.lo+total.hi)/2, 'energy_interval': total.as_list(),
                         **{f'E_{name}': (x.lo+x.hi)/2 for name, x in zip('fcrp', energies[u])},
                         'battery_initial': p['battery'], 'B_N': (B.lo+B.hi)/2, 'battery_interval': B.as_list(),
                         'endpoint_residual': float(residuals[u]),
                         'unknown_count': sum(s['conditions'][key]['status'] == 'unknown' for s in slot_results for key in ('C13', 'C14'))})
    for cid, names in FIELDS.items():
        if cid not in ('C07', 'C15', 'C17', 'C18'):
            for name in names: fields[name] = merge(fields[name], cs[cid])
    return {'layer': layer, 'conditions': cs, 'field_conditions': fields, 'slots': slot_results, 'tasks': task_rows, 'uavs': uav_rows,
            'diagnostics': diag, 'wall_time_s': perf_counter()-start}


def _replay_stream(instance, layer='evaluation', cfg=None):
    from .generator import accept
    from .queue import TaskState, inject, serve, advance_visit
    slots = []
    cfg = cfg or checker_config(); start = perf_counter()
    inst = dict(instance); inst['_geometry_accepted'] = accept(instance)['passed']
    p, tasks = inst['params'], inst['tasks']; N = round(p['T']/p['delta']); dt = p['delta']; eps = cfg['epsilon_num']
    cs = {key: verdict(0) for key in IDS}; states = [TaskState() for _ in tasks]
    positions = np.array(inst['qI']); velocities = np.array(inst['v0']); energies = [[iv(0) for _ in range(4)] for _ in range(p['U'])]
    route_node = [next((i for i, b in enumerate(inst['nodes']) if Box.from_dict(b).contains(q)), None) for q in positions]
    cs['C02'] = verdict(float(any(z is None for z in route_node)))
    current = [None]*p['U']; arrival = [0]*p['U']; arrival_flags = [True]*p['U']; caps = []
    visit_pointer = [0]*p['U']; slot_results = []; diag = {'rej_corridor_inner': 0, 'rej_service_inner': 0}
    cs['C03'] = verdict(0)
    fields = {f: verdict(0) for names in FIELDS.values() for f in names}
    payload = yield None
    while payload is not None:
        slot, checked = payload
        n = len(slots)
        slots.append(slot)
        if n == 0:
            cs['C10'] = verdict(float(np.max(np.linalg.norm(np.array(slots[0]['q'])-positions, axis=1))), eps)
            cs['C20'] = verdict(float(max(np.max(np.abs(np.array(slots[0]['v'])-velocities)), np.max(np.abs(slots[0]['W'])))), eps)
        cs['C09'] = merge(cs['C09'], verdict(float(slot['n'] != n)),
                          verdict(float(np.max(np.abs(positions-np.array(slot['q'])))), eps),
                          verdict(float(np.max(np.abs(velocities-np.array(slot['v'])))), eps))
        for u in range(p['U']):
            chosen = slot['edge'][u]; edge = inst['edges'][chosen]
            if n and current[u] is not None:
                prev = inst['edges'][current[u]]
                arrival_flags[u] = n > arrival[u] and Box.from_dict(inst['nodes'][prev['target']]).contains(positions[u], eps)
                if arrival_flags[u]: route_node[u], arrival[u] = prev['target'], n
            if arrival_flags[u]:
                cs['C01'] = merge(cs['C01'], verdict(float(edge['source'] != route_node[u])))
            else:
                cs['C01'] = merge(cs['C01'], verdict(float(chosen != current[u])))
            current[u] = chosen
        for k, state in enumerate(states):
            m = slot['locations'][k]
            release = state.events['g'].value
            if release == n:
                cs['C20'] = merge(cs['C20'], verdict(float(m is None)))
                state.location = m
            else:
                cs['C20'] = merge(cs['C20'], verdict(float(m != state.location)))
            if state.location is not None: inject(tasks[k], state, n)
            cs['C09'] = merge(cs['C09'], verdict(float(np.max(np.abs(state.W-np.array(slot['W'][k])))), max(eps, tasks[k]['C']*1e-12)))
        active = [tuple(x) for x in slot['active']]
        for u in range(p['U']):
            seq = inst['visits'][u]; idx = visit_pointer[u]
            own_active = [x for x in active if tasks[x[0]]['u'] == u]
            cs['C08'] = merge(cs['C08'], verdict(float(len(own_active) > 1 or any(idx >= len(seq) or list(x) != seq[idx] for x in own_active))))
            if idx < len(seq):
                k, phase = seq[idx]; st = states[k]
                if 0 < st.J[phase] < tasks[k]['L_'+phase]:
                    continuation = verdict(float((k, phase) not in active))
                    fields['viol_visit_continue'] = merge(fields['viol_visit_continue'], continuation)
                    cs['C07'] = merge(cs['C07'], continuation)
        for k, phase in active:
            st = states[k]
            if st.J[phase] >= tasks[k]['L_'+phase]: cs['C07'] = merge(cs['C07'], verdict(1)); continue
            advance_visit(st, phase, tasks[k]['L_'+phase], n)
            if st.J[phase] == tasks[k]['L_'+phase]: visit_pointer[tasks[k]['u']] += 1
        result = check_slot(inst, slot, layer, cfg) if checked is None else checked
        slot_results.append(result)
        for key in IDS: cs[key] = merge(cs[key], result['conditions'][key])
        for key, item in result['field_conditions'].items(): fields[key] = merge(fields[key], item)
        diag['rej_corridor_inner'] += result['rej_corridor_inner']
        mu, parts = slot_integrals(inst, slot, layer, cfg)
        caps.append([[iv(x) for x in row] for row in mu])
        guard_mu = np.array(slot['mu']); expected_mu, _ = slot_integrals(inst, slot, 'guard', cfg)
        cs['C09'] = merge(cs['C09'], verdict(float(np.max(np.abs(expected_mu-guard_mu))), max(eps, 1e-10*float(np.max(expected_mu)))))
        for k, st in enumerate(states): serve(st, guard_mu[k], n)
        for u in range(p['U']):
            for j in range(4): energies[u][j] += parts[u][j]
        positions, velocities = motion(positions, velocities, np.array(slot['accel']), dt)
        payload = yield result
    cs['C03'] = merge(verdict(float(len(slots) != N)), cs['C03'])
    if len(slots) == N:
        terminal_locations = slots[-1].get('terminal_locations', [st.location for st in states])
        for k, st in enumerate(states):
            if st.events['g'].value == N and st.location is None:
                st.location = terminal_locations[k]
                cs['C20'] = merge(cs['C20'], verdict(float(st.location not in range(p['M']+1))))
            else:
                cs['C20'] = merge(cs['C20'], verdict(float(terminal_locations[k] != st.location)))
            if st.location is not None: inject(tasks[k], st, N)
    residuals = np.linalg.norm(positions-np.array(inst['qF']), axis=1)
    cs['C10'] = merge(cs['C10'], verdict(float(max(residuals)), eps))
    for u, edgeidx in enumerate(current):
        if edgeidx is None or not Box.from_dict(inst['nodes'][inst['edges'][edgeidx]['target']]).contains(positions[u], eps):
            cs['C03'] = merge(cs['C03'], verdict(1))
    task_rows = []
    for k, t in enumerate(tasks):
        st = states[k]; ns, ng, na = (st.events[x].value for x in ('s', 'g', 'a'))
        lower = upper = ng
        stages = (1,) if st.location == 0 else (0, 1, 2)
        for stage in stages:
            work = [t['D'], t['C'], t['O']][stage]
            series = [c[k][stage] for c in caps]
            if stage == 1 and layer == 'audit':
                series = [Fraction(dt)*Fraction(s['f'][k]) for s in slots]
            lower = _completion(series, lower, work, True, layer != 'audit')
            upper = _completion(series, upper, work, False, layer != 'audit')
        if na is None: ready = deadline = fresh = None
        else:
            ready = 'pass' if upper is not None and upper <= na else 'fail' if lower is None or lower > na else 'unknown'
            deadline = 'pass' if (na+t['L_a'])*dt <= t['d'] else 'fail'
            fresh = 'pass' if ns is not None and (na-ns)*dt <= t['H'] else 'fail'
        window = [ready, deadline, fresh]
        cs['C18'] = merge(cs['C18'], *[{'status': x or 'fail', 'residual': 0. if x in ('pass', 'unknown') else 1.} for x in window])
        visits_done = all(st.J[x] == t['L_'+x] for x in ('s', 'a'))
        fields['viol_visit_complete'] = merge(fields['viol_visit_complete'], verdict(float(not visits_done)))
        cs['C07'] = merge(cs['C07'], fields['viol_visit_complete'])
        for name, status in zip(('viol_result_ready', 'viol_deadline', 'viol_freshness'), window):
            fields[name] = merge(fields[name], {'status': status or 'fail', 'residual': 0. if status in ('pass', 'unknown') else 1.})
        events = {x: e.as_dict() for x, e in st.events.items()}
        nr = lower if lower == upper else None
        events['r'] = {'available': nr is not None, 'value': nr}
        task_rows.append({'k': k, 'u': t['u'], 'started': ns is not None, 'completed': visits_done and window == ['pass']*3,
                          'events': events, 'n_s': ns, 'n_g': ng, 'n_r': nr, 'n_a': na,
                          'n_r_interval': [lower, upper], 'location': st.location, 'J_s': st.J['s'], 'J_a': st.J['a'],
                          'result_ready': ready, 'deadline': deadline, 'freshness': fresh,
                          'deadline_slack_s': None if na is None else t['d']-(na+t['L_a'])*dt,
                          'freshness_slack_s': None if na is None or ns is None else t['H']-(na-ns)*dt})
    uav_rows = []
    for u in range(p['U']):
        total = sum(energies[u], iv(0)); B = p['battery']-total
        status = 'pass' if B.lo >= 0 else 'fail' if B.hi < 0 else 'unknown'
        cs['C19'] = merge(cs['C19'], {'status': status, 'residual': -B.lo})
        uav_rows.append({'u': u, 'E_u': (total.lo+total.hi)/2, 'energy_interval': total.as_list(),
                         **{f'E_{name}': (x.lo+x.hi)/2 for name, x in zip('fcrp', energies[u])},
                         'battery_initial': p['battery'], 'B_N': (B.lo+B.hi)/2, 'battery_interval': B.as_list(),
                         'endpoint_residual': float(residuals[u]),
                         'unknown_count': sum(s['conditions'][key]['status'] == 'unknown' for s in slot_results for key in ('C13', 'C14'))})
    for cid, names in FIELDS.items():
        if cid not in ('C07', 'C15', 'C17', 'C18'):
            for name in names: fields[name] = merge(fields[name], cs[cid])
    return {'layer': layer, 'conditions': cs, 'field_conditions': fields, 'slots': slot_results, 'tasks': task_rows, 'uavs': uav_rows,
            'diagnostics': diag, 'wall_time_s': perf_counter()-start}


class ReplayAccumulator:
    def __init__(self, instance, cfg=None):
        self._stream = _replay_stream(instance, 'evaluation', cfg)
        self.wall_time_s = 0.
        self.result = None
        tic = perf_counter(); next(self._stream)
        self.wall_time_s += perf_counter()-tic

    def add(self, slot, checked=None):
        if self.result is not None: raise RuntimeError('evaluation already finished')
        tic = perf_counter()
        result = self._stream.send((slot, checked))
        self.wall_time_s += perf_counter()-tic
        return result

    def finish(self):
        if self.result is None:
            tic = perf_counter()
            try: self._stream.send(None)
            except StopIteration as stopped: self.result = stopped.value
            self.wall_time_s += perf_counter()-tic
            self.result['wall_time_s'] = self.wall_time_s
        return self.result
