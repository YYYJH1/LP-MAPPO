import heapq
import math
import hashlib
import json
import copy
from pathlib import Path
import numpy as np
from .geometry import Box
from .params import Params
from lawn_mec import FREEZE_ID
from lawn_mec.eval.oracle import necessary_bounds

VERSION = 'manhattan-fix2-reserve-1'
V7_VERSION = 'manhattan-v7-1'
V7B_VERSION = 'manhattan-v7-2'
V7C_VERSION = 'manhattan-v7-witness-1'
KNOB_DEFAULTS = {
    'reference_readiness': 'local', 'offload_ref_distance_m': 150.,
    'load_model': 'iid', 'load_normal': [0.1, 0.5], 'load_peak': [0.85, 0.98],
    'peak_dwell_s': 45., 'normal_dwell_s': 120., 'load_peak_fraction': [0.1, 0.4],
    'bs_placement': 'fixed', 'data_heterogeneity': None, 'reference_endpoint_fraction': 0.85,
    'cpb_heterogeneity': None, 'horizon_mode': 'fixed', 'horizon_fraction': .85,
    'battery_anchor': 'fixed', 'battery_margin': .05, 'battery_aggregate': 'max',
}


def generator_knobs(p):
    enabled = {}
    for name, choices in (('reference_readiness', ('local', 'neutral')),
                          ('load_model', ('iid', 'markov_peaks')),
                          ('bs_placement', ('fixed', 'random_block', 'random_corner')),
                          ('horizon_mode', ('fixed', 'reference')),
                          ('battery_anchor', ('fixed', 'free_flight', 'witness'))):
        value = p.get(name, choices[0])
        if value not in choices:
            raise ValueError(f'{name} must be one of {choices}')
        if value != choices[0]:
            enabled[name] = value
    def positive(name):
        value = p.get(name, KNOB_DEFAULTS[name])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value
    def interval(name, *, fractions=False):
        value = p.get(name, KNOB_DEFAULTS[name])
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
                or not 0 < value[0] <= value[1] or (fractions and value[1] >= 1)):
            raise ValueError(f'{name} needs finite 0 < lo <= hi' + (' < 1' if fractions else ''))
        return list(value)
    if 'reference_readiness' in enabled or 'horizon_mode' in enabled:
        enabled['offload_ref_distance_m'] = positive('offload_ref_distance_m')
    if 'horizon_mode' in enabled:
        enabled['reference_readiness'] = 'neutral'
        enabled['horizon_fraction'] = positive('horizon_fraction')
        if enabled['horizon_fraction'] > 1:
            raise ValueError('horizon_fraction must be <= 1')
    if 'battery_anchor' in enabled:
        margin = p.get('battery_margin', .10 if p.get('battery_anchor') == 'witness' else .05)
        if (isinstance(margin, bool) or not isinstance(margin, (int, float))
                or not math.isfinite(margin) or margin < 0):
            raise ValueError('battery_margin must be finite and nonnegative')
        aggregate = p.get('battery_aggregate', 'max')
        if aggregate not in ('max', 'mean'):
            raise ValueError('battery_aggregate must be max or mean')
        if p.get('battery_anchor') == 'witness' and aggregate != 'max':
            raise ValueError('witness battery requires the maximum per-UAV energy')
        enabled.update(battery_margin=margin, battery_aggregate=aggregate)
    if 'load_model' in enabled:
        for name in ('load_normal', 'load_peak', 'load_peak_fraction'):
            enabled[name] = interval(name, fractions=True)
        if enabled['load_normal'][1] >= enabled['load_peak'][0]:
            raise ValueError('load_normal and load_peak must be disjoint and ordered')
        for name in ('peak_dwell_s', 'normal_dwell_s'):
            enabled[name] = positive(name)
    if p.get('data_heterogeneity') is not None:
        enabled['data_heterogeneity'] = interval('data_heterogeneity')
    if p.get('cpb_heterogeneity') is not None:
        enabled['cpb_heterogeneity'] = interval('cpb_heterogeneity')
    fraction = positive('reference_endpoint_fraction')
    if fraction > 1:
        raise ValueError('reference_endpoint_fraction must be <= 1')
    if fraction != KNOB_DEFAULTS['reference_endpoint_fraction']:
        enabled['reference_endpoint_fraction'] = fraction
    return enabled


def corner_obstacle(rng, lo, hi, antenna, height, clearance, *, no_fly=False):
    for _ in range(10000):
        if no_fly:
            size = (hi-lo)/4-3
            lower = rng.uniform(lo, hi-size); upper = lower+size
        else:
            corners = rng.uniform(lo, hi, size=(2, 2))
            lower, upper = corners.min(axis=0), corners.max(axis=0)
            if np.any(upper-lower < 3):
                continue
        gap = np.maximum(np.maximum(lower-antenna, antenna-upper), 0)
        if np.linalg.norm(gap) >= clearance+1:
            return Box([*lower, 0], [*upper, height]).as_dict()
    raise ValueError('random_corner obstacle rejection budget exhausted')


def reference_horizon(instance, reference, cfg, cache):
    from lawn_mec.baselines.reference_schedule import reference_schedule
    p = dict(instance['params']); dt = p['delta']; N_max = round(p['T']/dt)
    if not math.isclose(N_max*dt, p['T'], rel_tol=0., abs_tol=1e-9):
        raise ValueError('reference horizon requires T_max / delta to be integral')
    n_ref = max(r['endpoint_slot'] for r in reference['uavs'].values())
    landing = 1+max(r['landing_slots'] for r in reference['uavs'].values())
    if n_ref+landing > N_max:
        return None
    fraction = p.get('horizon_fraction', .85)
    N_I = min(N_max, max(n_ref+landing, math.ceil(n_ref/fraction)))
    p['T'] = N_I*dt
    instance['params'] = p
    instance['load'] = [row[:N_I] for row in instance['load']]
    if 'load_provenance' in instance:
        meta = copy.deepcopy(instance['load_provenance'])
        for row in meta['bs']:
            row['segment_states'] = row['segment_states'][:math.ceil(N_I/meta['segment_slots'])]
        instance['load_provenance'] = meta
    plan = reference_schedule(instance, cfg, plan_cache=cache)
    if any(r['endpoint_slot'] > plan['endpoint_deadline_slot'] for r in plan['uavs'].values()):
        return None
    instance['horizon_reference'] = dict(n_ref=n_ref, f_T=fraction, N_I=N_I,
                                       N_max=N_max, landing_budget=landing,
                                       endpoint_after_truncation=max(r['endpoint_slot'] for r in plan['uavs'].values()))
    return plan


def free_flight_anchor(instance):
    from lawn_mec.baselines.reference_schedule import propulsion_operating_points
    from lawn_mec.env.model import shaft_power
    p = instance['params']; points = propulsion_operating_points(p)
    pmp = points['P_mp_shaft_W']; energies = []
    for visits in instance['visits']:
        energy = p['T']*(p['P0']+pmp/p['eta_flight'])
        for k, phase in visits:
            task = instance['tasks'][k]; duration = task['L_'+phase]*p['delta']
            psrv = shaft_power([task['V_'+phase], 0., 0.], np.zeros(3), p)
            energy += duration*(max(0., psrv-pmp)/p['eta_flight']+task['P_'+phase])
        energies.append(float(energy))
    aggregate = p.get('battery_aggregate', 'max'); margin = p.get('battery_margin', .05)
    p['battery'] = float((1+margin)*(max(energies) if aggregate == 'max' else np.mean(energies)))
    return {'kind': 'free_flight_anchor_not_bound', 'E_free_u': energies,
            'P_mp': pmp, 'V_mp': points['V_mp_mps'], 'P_srv': points['P_service_shaft_W'],
            'power_unit': 'shaft_W', 'energy_unit': 'J', 'aggregate': aggregate, 'margin': margin,
            'omits': ['climbs', 'transitions', 'descent_terms', 'CPU', 'radio']}


def markov_load(p, seed, n_bs, N):
    knobs = generator_knobs(p)
    segment = p['load_segment']; segment_s = segment*p['delta']
    cycle = knobs['peak_dwell_s']+knobs['normal_dwell_s']
    schedules, records = [], []
    for m in range(n_bs):
        rng = np.random.default_rng([seed, 701, m])
        q = float(rng.uniform(*knobs['load_peak_fraction']))
        normal_mean, peak_mean = (1-q)*cycle, q*cycle
        enter, leave = segment_s/normal_mean, segment_s/peak_mean
        if max(enter, leave) > 1:
            raise ValueError('load_segment is longer than a target-adjusted mean dwell')
        state = bool(rng.random() < q); states, fractions = [], []
        for _ in range((N+segment-1)//segment):
            states.append(int(state))
            fractions.append(float(rng.uniform(*knobs['load_peak' if state else 'load_normal'])))
            if rng.random() < (leave if state else enter):
                state = not state
        schedules.append((np.repeat(fractions, segment)[:N]*p['G_total']).tolist())
        records.append({'target_peak_fraction': q, 'normal_dwell_s': normal_mean,
                        'peak_dwell_s': peak_mean, 'normal_to_peak': enter,
                        'peak_to_normal': leave, 'segment_states': states})
    return schedules, {'model': 'markov_peaks', 'segment_slots': segment,
                       'dwell_rule': 'target_share_preserving_cycle_mean', 'bs': records}


def canonical(instance):
    return (json.dumps(instance, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def accept(instance):
    p = instance['params']; s = p['clearance']
    nodes = [Box.from_dict(b) for b in instance['nodes']]
    edges = [(e['source'], e['target'], Box.from_dict(e['box'])) for e in instance['edges']]
    checks = {name: True for name in ('edge_contains_nodes', 'self_loops', 'eroded_nonempty',
              'adjacent_overlap', 'eroded_connected', 'endpoints', 'endpoint_separation', 'service_nonempty', 'obstacle_clearance', 'bs_clearance')}
    for a, b, box in edges:
        checks['edge_contains_nodes'] &= all(box.contains(x) for n in (nodes[a], nodes[b]) for x in (n.lo, n.hi))
        if a == b: checks['self_loops'] &= bool(np.array_equal(box.lo, nodes[a].lo) and np.array_equal(box.hi, nodes[a].hi))
        checks['eroded_nonempty'] &= box.erode(s).nonempty
        for c, d, other in edges:
            if b == c:
                checks['adjacent_overlap'] &= box.erode(s).intersect(other.erode(s)).intersect(nodes[b].erode(s)).nonempty
        checks['obstacle_clearance'] &= all(box.distance(Box.from_dict(o)) >= s for o in instance['buildings']+instance['no_fly'])
        checks['bs_clearance'] &= all(box.point_distance(bs) >= s for bs in instance['bs'])
    checks['self_loops'] &= all(any(a == b == z for a, b, _ in edges) for z in range(len(nodes)))
    for start in range(len(nodes)):
        seen, todo = {start}, [start]
        while todo:
            z = todo.pop()
            for a, b, box in edges:
                if a == z and box.erode(s).nonempty and b not in seen:
                    seen.add(b); todo.append(b)
        checks['eroded_connected'] &= len(seen) == len(nodes)
    for u in range(p['U']):
        checks['endpoints'] &= all(any(b.erode(s).contains(instance[name][u]) for b in nodes) for name in ('qI', 'qF'))
    for name in ('qI', 'qF'):
        xy = np.asarray(instance[name])[:, :2]
        checks['endpoint_separation'] &= all(np.linalg.norm(xy[u]-xy[v]) >= p['D_h']+2-1e-9
                                             for u in range(p['U']) for v in range(u))
    for t in instance['tasks']:
        for phase in ('s', 'a'):
            srv = Box.from_dict(t['S_'+phase]); node = nodes[t['z_'+phase]].erode(s)
            checks['service_nonempty'] &= srv.intersect(node).nonempty and node.contains(srv.lo) and node.contains(srv.hi)
    return {'passed': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
            'reasons': [k for k, v in checks.items() if not v]}


def separated_endpoints(nodes, p, rng):
    order = rng.permutation(len(nodes)); chosen = []; spacing = p['D_h']+2
    candidates = [nodes[z].erode(p['clearance']).center for z in order] if p['U']<=len(nodes) else []
    grids=[]
    for z in order:
        b = nodes[z].erode(p['clearance'])
        grids.append([np.array([x,y,b.center[2]])
                      for x in np.arange(b.lo[0],b.hi[0]+1e-9,spacing)
                      for y in np.arange(b.lo[1],b.hi[1]+1e-9,spacing)])
    for i in range(max(map(len,grids),default=0)):
        candidates.extend(grid[i] for grid in grids if i<len(grid))
    for q in candidates:
        if all(np.linalg.norm(q[:2]-r[:2]) >= spacing-1e-9 for r in chosen):
            chosen.append(q)
            if len(chosen) == p['U']: return [q.tolist() for q in chosen]
    raise ValueError('endpoint_separation: insufficient eroded grid capacity')


def reference_times(instance):
    from lawn_mec.baselines.reference_schedule import reference_schedule
    p = instance['params']; dt = p['delta']
    plan_speed = p.get('V_ref', p['V_h'])
    plan = instance.get('reference')
    if plan is None:
        plan = reference_schedule(instance)
    boxes = [Box.from_dict(b).erode(p['clearance']) for b in instance['nodes']]
    c = np.array([b.center for b in boxes]); graph = {i: [] for i in range(len(c))}
    for e in instance['edges']:
        a,b = e['source'],e['target']; box = Box.from_dict(e['box']).erode(p['clearance'])
        if a != b and box.contains(c[a]) and box.contains(c[b]):
            graph[a].append((b,float(np.linalg.norm(c[b]-c[a]))))
    def leg(a,b):
        heap=[(0.,(a,))]; best={a:0.}
        while heap:
            length,path=heapq.heappop(heap); z=path[-1]
            if length > best[z]: continue
            if z==b: break
            for w,d in graph[z]:
                if length+d < best.get(w, math.inf):
                    best[w]=length+d; heapq.heappush(heap,(length+d,(*path,w)))
        else: raise ValueError('disconnected eroded reference graph')
        directions=[(c[y]-c[x])/np.linalg.norm(c[y]-c[x]) for x,y in zip(path,path[1:])]
        turns=sum(not np.allclose(x,y) for x,y in zip(directions,directions[1:]))
        duration=length/plan_speed+plan_speed/p['a_h']*(1+turns) if length else 0.
        return {'path':list(path),'length_m':length,'n_turn':turns,'travel_s':duration}
    result={'uavs':{},'tasks':{},'plan_speed':plan_speed}
    for u,visits in enumerate(instance['visits']):
        source=next(z for z,b in enumerate(boxes) if b.contains(instance['qI'][u]))
        planned=plan['uavs'][str(u)]
        clock_slot=plan['takeoff_slots']; last_cpu_end=0; rows=[]; task_rows={}
        for j,(k,phase) in enumerate(visits):
            t=instance['tasks'][k]; target=t['z_'+phase]; route=leg(source,target)
            travel_slots=planned['visits'][j]['travel_slots']; clock_slot+=travel_slots
            neutral = (p.get('reference_readiness', 'local') == 'neutral'
                       or p.get('horizon_mode', 'fixed') == 'reference')
            planned_task = plan['tasks'][str(k)] if neutral else None
            processing_slots=(planned_task['processing_s']/dt if neutral
                              else math.ceil(t['C']/(dt*p['F_local'])))
            r=task_rows.setdefault(k,{'processing_s': planned_task['processing_s'] if neutral else processing_slots*dt})
            if phase=='a': clock_slot=max(clock_slot,r['ready_slot'])
            r[phase+'_start_s']=clock_slot*dt
            rows.append({'visit':j,'k':k,'phase':phase,'start_s':clock_slot*dt,
                         **route,'scheduled_travel_s':travel_slots*dt})
            clock_slot+=t['L_'+phase]; source=target
            if phase=='s':
                if neutral:
                    cpu_start = planned_task['cpu_start_slot']
                    r.update({'reference_location': planned_task['reference_location'],
                              'cpu_start_s': None if cpu_start is None else cpu_start*dt,
                              'cpu_wait_s': planned_task['cpu_wait_slots']*dt,
                              'ready_slot': planned_task['ready_slot'],
                              'ready_s': planned_task['ready_slot']*dt})
                else:
                    cpu_start=max(clock_slot,last_cpu_end)
                    last_cpu_end=cpu_start+processing_slots
                    r.update({'cpu_start_s':cpu_start*dt,'cpu_wait_s':(cpu_start-clock_slot)*dt,
                              'ready_slot':last_cpu_end,'ready_s':last_cpu_end*dt})
        target=next(z for z,b in enumerate(boxes) if b.contains(instance['qF'][u]))
        last=leg(source,target)
        last['scheduled_travel_s']=planned['terminal_leg']['travel_slots']*dt
        clock=(clock_slot+planned['terminal_leg']['travel_slots'])*dt
        for k,r in task_rows.items():
            t=instance['tasks'][k]
            si=next(i for i,x in enumerate(rows) if x['k']==k and x['phase']=='s')
            ai=next(i for i,x in enumerate(rows) if x['k']==k and x['phase']=='a')
            between=sum(x['travel_s'] for x in rows[si+1:ai+1])+sum(instance['tasks'][x['k']]['L_'+x['phase']]*dt for x in rows[si+1:ai])
            r['sa_travel_service_s']=between
            r['compute_base_s']=r['cpu_wait_s']+r['processing_s']
            r['slack_base_s']=max(between,r['compute_base_s'])
            r['freshness_base_s']=t['L_s']*dt+r['slack_base_s']
            r['deadline_base_s']=r['s_start_s']+r['freshness_base_s']+t['L_a']*dt
            r['n_s_min']=r['s_start_s']/dt
            r['n_s_max']=(p['T']-(clock-r['s_start_s']))/dt
            result['tasks'][str(k)]=r
        travel_service=sum(row['travel_s']+instance['tasks'][row['k']]['L_'+row['phase']]*dt for row in rows)+last['travel_s']
        result['uavs'][str(u)]={'total_s':travel_service,'scheduled_total_s':clock,'visits':rows,'terminal_leg':last}
    return result


def reference_window_bounds(instance):
    from lawn_mec.baselines.reference_schedule import reference_schedule
    plan = instance.get('reference')
    if plan is None:
        plan = reference_schedule(instance)
    dt = instance['params']['delta']
    bounds = {}
    for uav in plan['uavs'].values():
        visits = uav['visits']
        positions = {(row['k'], row['phase']): j for j, row in enumerate(visits)}
        for row in visits:
            if row['phase'] != 's':
                continue
            k = row['k']; si = positions[k, 's']; ai = positions[k, 'a']
            b = dict(plan['tasks'][str(k)])
            between = (sum(v['travel_slots'] for v in visits[si+1:ai+1])
                       +sum(instance['tasks'][v['k']]['L_'+v['phase']]
                            for v in visits[si+1:ai]))*dt
            b['cpu_wait_s'] = b.get('cpu_wait_slots', 0)*dt
            b['sa_travel_service_s'] = between
            b['compute_base_s'] = b['cpu_wait_s']+b['processing_s']
            b['slack_base_s'] = max(between, b['compute_base_s'])
            bounds[str(k)] = b
    return bounds


def window_limits(task, bounds, dt, deadline_ratio, freshness_ratio, travel_ratio=1.0):
    compute = bounds['cpu_wait_s']+bounds['processing_s']
    between = bounds['sa_travel_service_s'] if travel_ratio == 1.0 else travel_ratio*bounds['sa_travel_service_s']
    freshness = task['L_s']*dt+max(between, freshness_ratio*compute)
    deadline = (bounds['s_start_s']+task['L_s']*dt
                +max(between, deadline_ratio*compute)+task['L_a']*dt)
    return deadline, freshness


def classify_tight_type(bounds, r_d, r_H, r_T=1.0):
    compute = bounds['cpu_wait_s']+bounds['processing_s']
    between = bounds['sa_travel_service_s'] if r_T == 1.0 else r_T*bounds['sa_travel_service_s']
    deadline_slack = max(between, r_d*compute)
    freshness_slack = max(between, r_H*compute)
    if deadline_slack < freshness_slack:
        return 'deadline'
    if freshness_slack < deadline_slack:
        return 'freshness'
    return 'travel'


def generate(params, seed):
    p = dict(params.data if isinstance(params, Params) else params)
    knobs = generator_knobs(p)
    if p.get('battery_anchor') == 'witness':
        p['battery'] = 1e9
        p.setdefault('battery_margin', .10)
    p.setdefault('V_ref', p['V_h'])
    if not math.isfinite(p['V_ref']) or not 0 < p['V_ref'] <= p['V_h']:
        raise ValueError('V_ref must be finite and satisfy 0 < V_ref <= V_h')
    from lawn_mec.baselines.reference_schedule import derive_descent_limit, reference_schedule
    from lawn_mec.baselines.common import config as baseline_config
    plan_config = baseline_config()
    p['v_desc_max'] = derive_descent_limit(p)
    rng = np.random.default_rng(seed)
    nodes = []
    for x in range(p['G_x']):
        for y in range(p['G_y']):
            nodes.append(Box([x*p['L_blk']-p['w_node']/2, y*p['L_blk']-p['w_node']/2, p['h_min']],
                             [x*p['L_blk']+p['w_node']/2, y*p['L_blk']+p['w_node']/2, p['h_max']]))
    edges = []
    for a, b in enumerate(nodes):
        for c, d in enumerate(nodes):
            if a == c or np.isclose(np.linalg.norm(b.center[:2]-d.center[:2]), p['L_blk']):
                edges.append({'source': a, 'target': c, 'box': Box(np.minimum(b.lo, d.lo), np.maximum(b.hi, d.hi)).as_dict()})
    buildings, no_fly, bs = [], [], []
    selected_blocks = None
    random_corner = p.get('bs_placement') == 'random_corner'
    if p.get('bs_placement', 'fixed') in ('random_block', 'random_corner'):
        blocks = (p['G_x']-1)*(p['G_y']-1)
        if p['M'] > blocks:
            raise ValueError('not enough blocks for requested BS count')
        selected_blocks = set(np.random.default_rng([seed, 702]).choice(blocks, size=p['M'], replace=False).tolist())
    for x in range(p['G_x']-1):
        for y in range(p['G_y']-1):
            center = np.array([(x+.5)*p['L_blk'], (y+.5)*p['L_blk']])
            margin = p['w_node']/2+p['clearance']+1
            lo = np.array([x*p['L_blk'], y*p['L_blk']])+margin
            hi = np.array([(x+1)*p['L_blk'], (y+1)*p['L_blk']])-margin
            block = x*(p['G_y']-1)+y
            corner = random_corner and block in selected_blocks
            antenna = np.random.default_rng([seed, 705, block]).uniform(lo, hi) if corner else lo+1
            obstacle_rng = np.random.default_rng([seed, 706, block]) if corner else None
            if ((selected_blocks is None and len(bs) < p['M']) or
                    (selected_blocks is not None and x*(p['G_y']-1)+y in selected_blocks)):
                bs.append([*antenna.tolist(), p['bs_height']])
            for _ in range(int(rng.integers(p['building_count'][0], p['building_count'][1]+1))):
                xy = rng.uniform(lo+3, (lo+hi)/2); upper = rng.uniform((lo+hi)/2, hi)
                height = rng.uniform(*p['building_height'])
                buildings.append(corner_obstacle(obstacle_rng, lo, hi, antenna, height, p['clearance'])
                                 if corner else Box([*xy, 0], [*upper, height]).as_dict())
            if rng.random() < p['no_fly_probability']:
                no_fly.append(corner_obstacle(obstacle_rng, lo, hi, antenna, p['h_max']+20,
                                             p['clearance'], no_fly=True) if corner else
                              Box([*(lo+3), 0], [*(lo+(hi-lo)/4), p['h_max']+20]).as_dict())
    if len(bs) < p['M']: raise ValueError('not enough blocks for requested BS count')
    N = round(p['T']/p['delta'])
    load = [[float(x) for x in np.repeat(rng.uniform(*p['load_fraction'], size=(N+p['load_segment']-1)//p['load_segment']),
                                       p['load_segment'])[:N]*p['G_total']] for _ in bs]
    load_provenance = None
    if p.get('load_model', 'iid') == 'markov_peaks':
        load, load_provenance = markov_load(p, seed, len(bs), N)
    data_rng = np.random.default_rng([seed, 703]) if p.get('data_heterogeneity') is not None else None
    cpb_rng = np.random.default_rng([seed, 704]) if p.get('cpb_heterogeneity') is not None else None
    variable_horizon = p.get('horizon_mode') == 'reference'
    v7b = random_corner or cpb_rng is not None or variable_horizon or p.get('battery_anchor') == 'free_flight'
    rejected = []; samples = []; accepted_uavs = {}; plan_cache = {}
    qI = separated_endpoints(nodes, p, rng); qF = separated_endpoints(nodes, p, rng)
    for attempt in range(p['max_resamples']):
        tasks, visits = [], []
        for u in range(p['U']):
            owned = []
            for _ in range(p['K_u']):
                k = len(tasks); owned.append(k)
                t = {'u': u, 'D': p['D'], 'C': p['D']/8*p['cycles_per_byte'], 'O': p['D']*p['output_ratio'],
                     'L_s': p['L_s'], 'L_a': p['L_a'], 'V_s': p['V_service'], 'V_a': p['V_service'],
                     'P_s': p['P_payload'], 'P_a': p['P_payload']}
                if data_rng is not None:
                    multiplier = float(data_rng.uniform(*p['data_heterogeneity']))
                    t.update(D=p['D']*multiplier, C=p['D']*multiplier/8*p['cycles_per_byte'],
                             O=p['D']*multiplier*p['output_ratio'], data_multiplier=multiplier)
                if cpb_rng is not None:
                    multiplier = float(np.exp(cpb_rng.uniform(*np.log(knobs['cpb_heterogeneity']))))
                    t.update(C=t['D']/8*p['cycles_per_byte']*multiplier, cpb_multiplier=multiplier)
                for phase in ('s', 'a'):
                    z = int(rng.integers(len(nodes))); center = nodes[z].center
                    half = np.array([p['w_srv']/2, p['w_srv']/2, (p['h_max']-p['h_min']-2*p['clearance'])*p['service_height_fraction']/2])
                    t['z_'+phase], t['S_'+phase] = z, Box(center-half, center+half).as_dict()
                tasks.append(t)
            remaining = list(rng.permutation(owned).astype(int)); pending, seq = [], []
            while remaining or pending:
                if remaining and (not pending or rng.random() < p['interleave']):
                    k = int(remaining.pop(0)); seq.append([k, 's']); pending.append(k)
                else:
                    k = pending.pop(int(rng.integers(len(pending)))); seq.append([k, 'a'])
            visits.append(seq)
        inst = {'generator_version': VERSION, 'freeze_id': FREEZE_ID, 'seed': int(seed), 'params': p,
                'nodes': [b.as_dict() for b in nodes], 'edges': edges, 'buildings': buildings, 'no_fly': no_fly,
                'bs': bs, 'load': load, 'tasks': tasks, 'visits': visits, 'qI': qI, 'qF': qF,
                'v0': [[0., 0., 0.] for _ in range(p['U'])], 'state': 'undetermined'}
        if knobs:
            inst.update(generator_version=V7B_VERSION if v7b else V7_VERSION, generator_knobs=knobs)
        if load_provenance is not None:
            inst['load_provenance'] = load_provenance
        for u, saved in accepted_uavs.items():
            inst['qI'][u],inst['qF'][u],inst['visits'][u],saved_tasks = saved
            tasks[u*p['K_u']:(u+1)*p['K_u']] = saved_tasks
        reference = reference_schedule(inst, plan_config, plan_cache=plan_cache)
        sample = {'attempt': attempt+1,
                  'planned_endpoint_max': max(r['endpoint_slot'] for r in reference['uavs'].values()),
                  'rejected_reasons': [], 'released_uavs': 0}
        if v7b:
            sample['route_length_m_per_uav'] = [float(sum(
                np.linalg.norm(nodes[b].center-nodes[a].center)
                for leg in [*r['visits'], r['terminal_leg']]
                for a, b in zip(leg['path'], leg['path'][1:]))) for r in reference['uavs'].values()]
        samples.append(sample)
        landing_budget = 1+max(r['landing_slots'] for r in reference['uavs'].values())
        endpoint_limit = (min(N-landing_budget, reference['endpoint_deadline_slot']) if variable_horizon else
                          min(p.get('reference_endpoint_fraction', .85)*N, reference['endpoint_deadline_slot']))
        for u in range(p['U']):
            if reference['uavs'][str(u)]['horizontal_end_slot']+landing_budget <= endpoint_limit:
                accepted_uavs[u] = (qI[u],qF[u],visits[u],tasks[u*p['K_u']:(u+1)*p['K_u']])
        if any(r['endpoint_slot'] > endpoint_limit for r in reference['uavs'].values()):
            sample['rejected_reasons'] = ['reference_horizon']
            rejected.append(sample['rejected_reasons']); continue
        if variable_horizon:
            reference = reference_horizon(inst, reference, plan_config, plan_cache)
            if reference is None:
                sample['rejected_reasons'] = ['reference_horizon_after_truncation']
                accepted_uavs.clear()
                rejected.append(sample['rejected_reasons']); continue
        inst['reference'] = reference
        inst['reference_analytic'] = reference_times(inst)
        bounds = reference_window_bounds(inst)
        for k, t in enumerate(tasks):
            b = bounds[str(k)]
            rd = rng.uniform(*p['window_deadline_ratio'])
            rh = rng.uniform(*p['window_freshness_ratio'])
            rt = rng.uniform(*p['window_travel_ratio']) if p.get('window_travel_ratio') is not None else 1.0
            t['d'], t['H'] = window_limits(t, b, p['delta'], rd, rh, rt)
            t['window_ratios'] = {'deadline': rd, 'freshness': rh} if rt == 1.0 and p.get('window_travel_ratio') is None else {'deadline': rd, 'freshness': rh, 'travel': rt}
            switch = (t['d']-t['L_a']*p['delta']-t['H'])/p['delta']
            t['tight_type'] = classify_tight_type(b, rd, rh, rt)
            t['tight_switch_slot'], t['window_bounds'] = switch, b
        acceptance = accept(inst)
        if not acceptance['passed']:
            sample['rejected_reasons'] = acceptance['reasons']
            rejected.append(sample['rejected_reasons']); continue
        if not {'deadline', 'freshness'} <= {t['tight_type'] for t in tasks}:
            sample['rejected_reasons'] = ['tight_type_mixture']
            travel_uavs = {t['u'] for t in tasks if t['tight_type'] == 'travel'}
            for u in sorted(travel_uavs):
                if u in accepted_uavs:
                    accepted_uavs.pop(u)
                    sample['released_uavs'] += 1
            rejected.append(sample['rejected_reasons']); continue
        inst['acceptance'] = acceptance
        if p.get('battery_anchor') == 'free_flight':
            inst['params'] = dict(inst['params'])
            inst['battery_anchor'] = free_flight_anchor(inst)
        inst['resampling'] = {'attempt': attempt+1, 'rejected_reasons': rejected, 'samples': samples,
                              'acceptance_rate': 1/(attempt+1)}
        inst['oracle'] = necessary_bounds(inst)
        inst['state'] = 'infeasible_proven' if inst['oracle']['infeasible_proven'] else 'undetermined'
        if p.get('battery_anchor') == 'witness':
            inst['generator_version'] = V7C_VERSION
            inst['certification'] = {'status': 'pending', 'placeholder_battery_J': 1e9,
                                     'placeholder_used': True, 'planner_required': 'certify_pool_v7'}
        return inst
    raise ValueError(f'instance discarded after {p["max_resamples"]} attempts: {rejected}')


def sampling_summary(instances):
    instances = list(instances)
    samples = [s for inst in instances for s in inst['resampling'].get('samples', [])]
    attempts = sum(inst['resampling']['attempt'] for inst in instances)
    def quantiles(rejected):
        values = [s['planned_endpoint_max'] for s in samples if bool(s['rejected_reasons']) == rejected]
        return {'count': len(values), 'quantiles': dict(zip(('10', '50', '90'),
                np.quantile(values, [.1, .5, .9]).tolist() if values else [None]*3))}
    return {'instances': len(instances), 'attempts': attempts, 'resamples': attempts-len(instances),
            'acceptance_rate': len(instances)/attempts if attempts else None,
            'recorded_attempts': len(samples), 'endpoint_unit': 'slot',
            'rejected_planned_endpoint_max': quantiles(True),
            'accepted_planned_endpoint_max': quantiles(False)}


def save_instance(instance, directory):
    payload = canonical(instance); digest = hashlib.sha256(payload).hexdigest()
    path = Path(directory)/f'{digest[:16]}.json'; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload: raise RuntimeError('hash prefix collision')
    path.write_bytes(payload)
    return path
