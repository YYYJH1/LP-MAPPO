import math
import numpy as np
from lawn_mec.env.geometry import Box
from lawn_mec.env.model import power_floor


def necessary_bounds(instance, locations=None):
    p, tasks = instance['params'], instance['tasks']
    dt, N = p['delta'], round(p['T']/p['delta'])
    vmax = math.hypot(p['V_h'], max(p['V_up'], p['V_down']))
    G = p['G_total']-np.array(instance['load'])
    regions = [Box.from_dict(e['box']).erode(p['clearance']) for e in instance['edges']]
    link = []
    for bs in instance['bs']:
        d = min(b.point_distance(bs) for b in regions)
        g = p['beta_L']*(d/p['d0'])**(-p['alpha_L'])
        link.append([p['bandwidth']*math.log2(1+p[power]*g/(p['N0']*p['bandwidth'])) for power in ('P_ul', 'P_dl')])
    def slots(work, capacity):
        return max(1, math.ceil(work/capacity)) if capacity > 0 else math.inf
    processing, per_bs = {}, {}
    for k, t in enumerate(tasks):
        local = slots(t['C'], dt*p['F_local'])
        options = [[slots(t['D'], dt*r[0]), slots(t['C'], dt*max(G[m])), slots(t['O'], dt*r[1])]
                   for m, r in enumerate(link)]
        per_bs[k] = options
        processing[k] = min([local]+[sum(x) for x in options]) if locations is None else (
            local if locations[k] == 0 else sum(options[locations[k]-1]))
    task_bounds, uav_bounds, reasons = {}, {}, []
    for u, visits in enumerate(instance['visits']):
        prev = Box(instance['qI'][u], instance['qI'][u])
        earliest, starts = 0., {}
        for k, phase in visits:
            region = Box.from_dict(tasks[k]['S_'+phase])
            earliest += prev.distance(region)/vmax
            starts[(k, phase)] = earliest
            earliest += tasks[k]['L_'+phase]*dt
            prev = region
        horizon = earliest+prev.point_distance(instance['qF'][u])/vmax
        floor_energy = p['T']*(p['P0']+power_floor(p)/p['eta_flight'])+sum(
            t['P_s']*t['L_s']*dt+t['P_a']*t['L_a']*dt for t in tasks if t['u'] == u)
        uav_bounds[str(u)] = {'horizon_s': horizon, 'energy_J': floor_energy}
        if horizon > p['T']: reasons.append(f'horizon:u{u}')
        if floor_energy > p['battery']: reasons.append(f'energy:u{u}')
        for k, t in enumerate(tasks):
            if t['u'] != u: continue
            between = starts[(k, 'a')]-starts[(k, 's')]-t['L_s']*dt
            sa = t['L_s']*dt+max(between, processing[k]*dt)
            ns = math.ceil(starts[(k, 's')]/dt-1e-12)
            finish = ns*dt+sa+t['L_a']*dt
            task_bounds[str(k)] = {'n_s_min': ns, 't_sa_min': sa, 't_f_min': finish,
                                   'processing_slots': processing[k],
                                   'n_s_max': N-t['L_s']-processing[k]-t['L_a']}
            if t.get('H', math.inf) < sa: reasons.append(f'freshness:k{k}')
            if t.get('d', math.inf) < finish: reasons.append(f'deadline:k{k}')
    if locations is None:
        if sum(t['C'] for t in tasks) > p['U']*N*dt*p['F_local']+dt*G.sum():
            reasons.append('compute:total_supply')
    else:
        if len(locations) != len(tasks) or any(m not in range(p['M']+1) for m in locations):
            raise ValueError('one valid location is required per task')
        for u in range(p['U']):
            local = [k for k, t in enumerate(tasks) if t['u'] == u and locations[k] == 0]
            remote = [k for k, t in enumerate(tasks) if t['u'] == u and locations[k] > 0]
            if sum(tasks[k]['C'] for k in local) > N*dt*p['F_local']: reasons.append(f'compute:u{u}')
            if sum(per_bs[k][locations[k]-1][0]+per_bs[k][locations[k]-1][2] for k in remote) > N:
                reasons.append(f'tdma:u{u}')
        for m in range(1, p['M']+1):
            ks = [k for k in range(len(tasks)) if locations[k] == m]
            if sum(tasks[k]['C'] for k in ks) > dt*G[m-1].sum(): reasons.append(f'compute:bs{m}')
            if sum(per_bs[k][m-1][0]+per_bs[k][m-1][2] for k in ks) > N: reasons.append(f'tdma:bs{m}')
    return {'infeasible_proven': bool(reasons), 'reasons': reasons,
            'bounds': {'tasks': task_bounds, 'uavs': uav_bounds, 'R_max': link, 'V_max_3D': vmax,
                       'per_bs_slots': {str(k): v for k, v in per_bs.items()}}}


def verify_witness(trajectory):
    from lawn_mec.env.checker import replay_episode
    from .registry import aggregate
    result = replay_episode(trajectory['instance'], trajectory['slots'], layer='audit')
    status = aggregate(result['conditions'])['feasible_status']
    necessary = necessary_bounds(trajectory['instance'])
    state = 'complete_witness' if status == 'p0' else 'infeasible_proven' if necessary['infeasible_proven'] else 'undetermined'
    return {'state': state, 'necessary_conditions': necessary,
            'witness_status': 'pass' if status == 'p0' else ('unknown' if status == 'unknown' else 'fail'),
            'audit': result}
