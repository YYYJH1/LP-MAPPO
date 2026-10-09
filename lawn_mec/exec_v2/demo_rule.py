import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from lawn_mec.env.checker import replay_episode
from lawn_mec.env.geometry import Box
from lawn_mec.env.model import motion
from lawn_mec.env.queue import inject
from .state import SimState, clone_task
from .step import SlotActions, step_v2
from .primitives import PrimitiveLibrary, Uncertified
from .motion import MotionPlanner, Target


def node_of(instance, point):
    return next(i for i, b in enumerate(instance['nodes']) if Box.from_dict(b).contains(point))


def schedule(instance, library=None, planner=None):
    lib = library or PrimitiveLibrary(instance)
    planner = planner or MotionPlanner(lib)
    U = instance['params']['U']; N = round(instance['params']['T']/instance['params']['delta'])
    if U > len(lib.heights):
        raise Uncertified('exclusive layer count exceeded')
    takeoffs = []
    for u in range(U):
        node = node_of(instance, instance['qI'][u])
        if np.linalg.norm(np.array(instance['v0'][u])) > lib.cfg['epsilon_num']:
            raise Uncertified('demo requires initial rest')
        if np.linalg.norm(np.array(instance['qI'][u])-lib.point(node, 85.)) > lib.cfg['epsilon_num']:
            raise Uncertified('demo requires initial node centre at 85 m')
        takeoffs.append(min(lib.vertical(node, 85., lib.heights[u]), key=lambda x: (x.duration, x.E, x.name)))
    takeoff_end = max(p.duration for p in takeoffs)
    schedules, starts, timings = [], [], []
    for u in range(U):
        height = lib.heights[u]; node = node_of(instance, instance['qI'][u])
        chain = [takeoffs[u]]+[lib.hover(node, height)]*(takeoff_end-takeoffs[u].duration)
        n = takeoff_end; q = lib.point(node, height); start_flags = {}
        for k, phase in instance['visits'][u]:
            dest = instance['tasks'][k]['z_'+phase]
            T = n+planner.earliest_duration(node, dest, height)
            if T+instance['tasks'][k]['L_'+phase] > planner.landing_start:
                raise Uncertified(f'demo UAV {u}, visit {(k, phase)} cannot fit before landing')
            plan = planner.plan(q, np.zeros(3), n, Target('service', task=k, phase=phase), T, height)
            timings.append(plan.call_time_s)
            if plan.status != 'ok':
                raise Uncertified(f'demo earliest target failed: {plan}')
            chain.extend(plan.chain); start_flags[T] = (k, phase)
            service = min(lib.service(k, phase, height), key=lambda x: (x.E, x.name))
            chain.append(service); n = T+service.duration; node = dest; q = lib.point(node, height)
        plan = planner.plan(q, np.zeros(3), n, Target('endpoint', uav=u), N, height)
        timings.append(plan.call_time_s)
        if plan.status != 'ok':
            raise Uncertified(f'demo UAV {u} cannot return before common landing: {plan.earliest_slot}')
        chain.extend(plan.chain)
        rows = [dict(edge=e, accel=list(a), family=p.family, primitive=p.name)
                for p in chain for e, a in zip(p.edges, p.accelerations)]
        if len(rows) != N:
            raise AssertionError('planner did not cover the horizon exactly')
        schedules.append(rows); starts.append(start_flags)
    return schedules, starts, dict(planner_calls_s=timings, takeoff_end=takeoff_end,
                                  landing_start=planner.landing_start, layers=list(lib.heights[:U]))


def run(instance, *, legacy=True, library=None, planner=None):
    tic = perf_counter()
    schedules, starts, timing = schedule(instance, library, planner)
    state = SimState(instance)
    U, K = state.p['U'], len(state.tasks)
    step_times = []
    while state.n < state.N:
        n = state.n
        locations = [0 if st.events['g'].value == n else st.location for st in state.states]
        pending = [clone_task(st) for st in state.states]
        for k, st in enumerate(pending):
            st.location = locations[k]
            if st.location is not None:
                inject(state.tasks[k], st, n)
        f = np.zeros(K)
        for u in range(U):
            available = [k for k, st in enumerate(pending) if state.tasks[k]['u'] == u and st.location == 0 and st.W[1] > 0]
            if available:
                k = min(available, key=lambda k: (pending[k].events['g'].value, k))
                f[k] = state.p['F_local']
        actions = SlotActions([schedules[u][n]['edge'] for u in range(U)],
                              [schedules[u][n]['accel'] for u in range(U)],
                              [n in starts[u] for u in range(U)], locations,
                              np.zeros((K, 2)), np.zeros(K), f)
        begin = perf_counter()
        step_v2(state, actions, on_infeasible='record')
        step_times.append(perf_counter()-begin)
    logs = state.logs()
    audit = replay_episode(instance, logs, 'audit')
    groups = {name: 0. for name in ('loiter', 'hover', 'cruise', 'service', 'vertical')}
    flight_groups = groups.copy()
    from lawn_mec.env.checker import slot_integrals
    for n, record in enumerate(logs):
        _, intervals = slot_integrals(state.instance, record, 'audit', state.cfg)
        energy = [[(x.lo+x.hi)/2 for x in row] for row in intervals]
        for u in range(U):
            family = schedules[u][n]['family']
            groups[family] += sum(energy[u]); flight_groups[family] += energy[u][0]
    result = dict(seed=instance.get('seed'), conditions=audit['conditions'], C=sum(t['completed'] for t in audit['tasks']),
                  energies_J=[u['E_u'] for u in audit['uavs']], energy_intervals_J=[u['energy_interval'] for u in audit['uavs']],
                  energy_by_family_J=groups, energy_share={k: v/sum(groups.values()) for k, v in groups.items()},
                  flight_share={k: v/sum(flight_groups.values()) for k, v in flight_groups.items()},
                  evaluation_flagged_slots=sum(row['exec_v2_status'] != 'ok' for row in logs),
                  timings=dict(**timing, step_median_ms=float(np.median(step_times))*1000,
                               step_p95_ms=float(np.percentile(step_times, 95))*1000))
    if legacy:
        from lawn_mec.baselines.policies import BaselinePolicy, rollout
        env, _, _ = rollout(instance, BaselinePolicy(instance, 'B1'))
        old = replay_episode(instance, env.logs, 'audit')
        result['legacy_B1'] = dict(C=sum(t['completed'] for t in old['tasks']),
                                   energies_J=[u['E_u'] for u in old['uavs']], conditions=old['conditions'])
        result['energy_change_fraction'] = sum(result['energies_J'])/sum(result['legacy_B1']['energies_J'])-1
    result['wall_time_s'] = perf_counter()-tic
    return result, logs, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    allowed = Path(os.environ.get('LPM_RUNS_ROOT', Path(__file__).resolve().parents[2]/'runs')).resolve()/'exec_v2'
    if allowed not in args.out.resolve().parents:
        raise ValueError(f'demo outputs must be inside {allowed}')
    result, logs, audit = run(json.loads(args.instance.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (('result', result), ('logs', logs), ('audit', audit)):
        (args.out/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
