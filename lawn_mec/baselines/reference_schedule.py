from functools import lru_cache
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from lawn_mec.env.model import shaft_power, power_floor


def propulsion_operating_points(p):
    from lawn_mec.env.model import shaft_parameters
    return dict(_operating_points(shaft_parameters(p, ('V_h', 'V_service', 'P0', 'eta_flight',
                                                       'power_floor_ratio'))))


@lru_cache(maxsize=32)
def _operating_points(items):
    p = dict(items)
    def shaft(v):
        return shaft_power([v, 0., 0.], np.zeros(3), p)
    def electrical(power):
        return p['P0']+power/p['eta_flight']
    def minimum(fn, lower):
        grid = np.linspace(lower, p['V_h'], 513)
        values = [fn(v) for v in grid]
        candidates = [(values[0], float(grid[0])), (values[-1], float(grid[-1]))]
        for j in range(1, len(grid)-1):
            if values[j] <= min(values[j-1], values[j+1]):
                result = minimize_scalar(fn, bounds=(grid[j-1], grid[j+1]), method='bounded',
                                         options={'xatol': 1e-11})
                candidates.append((float(result.fun), float(result.x)))
        return min(candidates)
    pmp, vmp = minimum(shaft, 0.)
    _, vmr = minimum(lambda v: electrical(shaft(v))/v, 1e-8)
    srv = shaft(p['V_service']); floor = power_floor(p)
    loiter = shaft_power([6., 0., 0.], [0., 6.**2/12., 0.], p)
    return tuple(dict(V_mp_mps=vmp, P_mp_shaft_W=pmp, V_mr_mps=vmr,
                      P_hover_W=electrical(shaft(0.)), P_mp_W=electrical(pmp),
                      V_service_mps=p['V_service'], P_service_shaft_W=srv,
                      P_service_W=electrical(srv), P_loiter_r12_v6_W=electrical(loiter),
                      power_floor_shaft_W=floor, power_floor_W=electrical(floor),
                      derive_descent_limit_mps=derive_descent_limit(p)).items())


def derive_descent_limit(p):
    keys = ('V_h', 'weight', 'g0', 'rho', 'drag_area', 'rotor_area',
            'thrust_coeff', 'profile_delta', 'solidity', 'induced_epsilon',
            'power_floor_ratio')
    return _descent_bound(tuple((k, float(p[k])) for k in keys))


@lru_cache(maxsize=32)
def _descent_bound(items):
    p = dict(items)
    grid = np.linspace(0., p['V_h'], 257)
    def margin(descent):
        def power(h): return shaft_power([h, 0., -descent], np.zeros(3), p)
        values = np.array([power(h) for h in grid])
        candidates = [values[0], values[-1]]
        for j in range(1, len(grid)-1):
            if values[j] <= min(values[j-1], values[j+1]):
                result = minimize_scalar(power, bounds=(grid[j-1], grid[j+1]),
                                         method='bounded', options={'xatol': 1e-12})
                candidates.append(result.fun)
        return min(candidates)-power_floor(p)-1.
    if margin(0.) <= 0: raise ValueError('no descent power margin at zero descent')
    upper = 1.
    while margin(upper) > 0:
        upper *= 2
        if upper > 128: raise ValueError('descent power root not bracketed')
    return float(brentq(margin, 0., upper, xtol=1e-12))


def offload_reference_slots(instance):
    p = instance['params']; dt = p['delta']
    distance = p.get('offload_ref_distance_m', 150.)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('offload_ref_distance_m must be finite and positive')
    gain = p['beta_L']*(distance/p['d0'])**(-p['alpha_L'])
    rates = [p['bandwidth']*math.log2(1+p[key]*gain/(p['N0']*p['bandwidth']))
             for key in ('P_ul', 'P_dl')]
    idle = float(np.mean(p['G_total']-np.asarray(instance['load'], dtype=float)))
    def slots(work, capacity):
        return max(1, math.ceil(work/capacity)) if capacity > 0 else math.inf
    return [sum((slots(t['D'], dt*rates[0]), slots(t['C'], dt*idle),
                 slots(t['O'], dt*rates[1]))) for t in instance['tasks']]


def reference_schedule(instance, cfg=None, *, plan_cache=None):
    from .common import config
    from .policies import Navigator
    p = instance['params']; dt = p['delta']; N = round(p['T']/dt)
    readiness = p.get('reference_readiness', 'local')
    if readiness not in ('local', 'neutral'):
        raise ValueError('reference_readiness must be local or neutral')
    neutral = readiness == 'neutral' or p.get('horizon_mode', 'fixed') == 'reference'
    offload = offload_reference_slots(instance) if neutral else None
    plan_speed = p.get('V_ref', p['V_h'])
    if not math.isfinite(plan_speed) or not 0 < plan_speed <= p['V_h']:
        raise ValueError('V_ref must be finite and satisfy 0 < V_ref <= V_h')
    plan_cfg = dict(cfg or config())
    plan_cfg.update(speed_scale=plan_speed/p['V_h'],
                    corner_speed=min(plan_cfg['corner_speed'], plan_speed))
    cache = None if plan_cache is None else plan_cache.setdefault(('reference_speed', plan_speed), {})
    nav = Navigator(instance, 'B1', plan_cfg, plan_only=True, plan_cache=cache)
    result = {'kind': 'B1_navigator_plan', 'plan_speed': plan_speed, 'takeoff_slots': nav.takeoff_slots,
              'n_reserve': nav.cfg['n_reserve'], 'endpoint_deadline_slot': nav.endpoint_deadline,
              'uavs': {}, 'tasks': {}}
    if neutral:
        result['kind'] = 'neutral_navigator_plan'
    horizontal_ends = []
    for u, visits in enumerate(instance['visits']):
        clock = nav.takeoff_slots; rows = []; ready = {}; last_cpu_end = 0
        for j, (k, phase) in enumerate(visits):
            t = instance['tasks'][k]
            travel = sum(len(s['accels']) for s in nav.legs[u][j])
            clock += travel
            if phase == 'a': clock = max(clock, ready[k])
            rows.append({'visit': j, 'k': k, 'phase': phase, 'start_slot': clock,
                         'start_s': clock*dt, 'travel_slots': travel, 'path': nav.paths[u][j]})
            task = result['tasks'].setdefault(str(k), {})
            task[phase+'_start_slot'] = clock; task[phase+'_start_s'] = clock*dt
            clock += t['L_'+phase]
            if phase == 's':
                cpu_start = max(clock, last_cpu_end)
                local_slots = math.ceil(t['C']/(dt*p['F_local']))
                if neutral and clock+offload[k] < cpu_start+local_slots:
                    ready[k] = clock+offload[k]
                    task.update({'reference_location': 'offload', 'processing_s': offload[k]*dt,
                                 'ready_slot': ready[k], 'cpu_start_slot': None, 'cpu_wait_slots': 0})
                else:
                    ready[k] = cpu_start+local_slots; last_cpu_end = ready[k]
                    task.update({'processing_s': local_slots*dt, 'ready_slot': ready[k],
                                 'cpu_start_slot': cpu_start, 'cpu_wait_slots': cpu_start-clock})
                    if neutral:
                        task['reference_location'] = 'local'
        terminal_slots = sum(len(s['accels']) for s in nav.legs[u][-1])
        clock += terminal_slots; horizontal_ends.append(clock)
        result['uavs'][str(u)] = {'visits': rows, 'horizontal_end_slot': clock,
                                  'terminal_leg': {'travel_slots': terminal_slots, 'path': nav.paths[u][-1]},
                                  'landing_slots': len(nav.landing[u])}
    landing_start = max(horizontal_ends)+1
    result['landing_start_slot'] = landing_start
    for u, r in result['uavs'].items():
        end = landing_start+r['landing_slots'] if r['landing_slots'] else r['horizontal_end_slot']
        r.update({'endpoint_slot': end, 'total_s': end*dt, 'scheduled_total_s': end*dt})
        for row in r['visits']:
            k = row['k']; t = instance['tasks'][k]; task = result['tasks'][str(k)]
            task.update({'freshness_base_s': (task['a_start_slot']-task['s_start_slot'])*dt,
                         'deadline_base_s': (task['a_start_slot']+t['L_a'])*dt,
                         'n_s_min': task['s_start_slot'], 'n_s_max': N-end+task['s_start_slot']})
    return result
