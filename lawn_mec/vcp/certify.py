from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math
import signal
import time
import numpy as np

from lawn_mec.exec_v2.certification import anchor_instance, summary_audit
from lawn_mec.exec_v2.events import EventExecutor
from lawn_mec.exec_v2.motion import MotionPlanner
from lawn_mec.exec_v2.primitives import PrimitiveLibrary, Uncertified
from lawn_mec.exec_v2.policies import METHODS, rollout as el_rollout, slack_priorities
from lawn_mec.exec_v2.search import objective, feasible, certified_energies, digest, moves
from lawn_mec.exec_v2.p0_loss import hard_nonpass
from .dp_commit import RelaxedDP
from .executor import CommitmentPolicy, SuffixExecutor, VARIANTS, rollout
from .search import local_search, readiness_counts
from .tables import CommitmentTables

VERSION = 'ls-el+vcp-1'


class CpuLimit(BaseException):
    pass


def is_cpu_limit(exc):
    seen = set()
    while isinstance(exc, SystemError) and id(exc) not in seen:
        seen.add(id(exc))
        exc = exc.__cause__
    return isinstance(exc, CpuLimit)


class CommitmentLimit(BaseException):
    pass


@contextmanager
def deadline(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite CPU seconds required')
    if signal.getitimer(signal.ITIMER_PROF) != (0., 0.):
        raise RuntimeError('nested CPU timers are unsupported')
    previous = signal.getsignal(signal.SIGPROF)
    def expire(signum, frame):
        raise CpuLimit()
    signal.signal(signal.SIGPROF, expire)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.)
        signal.signal(signal.SIGPROF, previous)


def realized_plan(result, instance):
    plan = dict(layers=None, locations={}, offsets={}, routes={}, truncations={}, deferrals={})
    earliest = {}
    for event in result['decisions']:
        choices = event['choices']
        if event['kind'] == 'layers':
            plan['layers'] = list(choices['layers'])
        elif event['kind'] == 'locations':
            plan['locations'].update({str(k): v for k, v in choices['locations'].items()})
        elif event['kind'] == 'commitments':
            for req in event['requests']:
                key = f"{req['uav']}:{req['pointer']}"
                choice = choices['commitments'][req['uav']]
                if choice.get('truncate'):
                    plan['truncations'][key] = choice['route']
                    continue
                if req['reason'] == 'deferral':
                    plan['deferrals'][f"{key}:{event['slot']}"] = deepcopy(choice)
                    continue
                if choice['route'] == 'keep':
                    continue
                plan['routes'][key] = choice['route']
                earliest[key] = req['routes'][choice['route']]['earliest']
                plan['offsets'][key] = choice['T']-earliest[key]
    return plan


def plan_rollout(instance, plan, tables, commitment=None, variant='earliest_feasible', *,
                 execution_layer='evaluation'):
    ex = SuffixExecutor(instance, library=tables.lib, planner=tables.planner,
                        execution_layer=execution_layer)
    policy = CommitmentPolicy(commitment, tables, variant) if commitment is not None else None
    while (event := ex.next_event())['kind'] != 'done':
        kind = event['kind']
        if kind == 'layers':
            decision = dict(layers=plan['layers'])
        elif kind == 'locations':
            decision = dict(locations={k: plan['locations'][str(k)] for k in event['tasks']})
        elif kind == 'priorities':
            decision = dict(priorities=policy.priorities(ex, event) if policy else slack_priorities(ex).tolist())
        else:
            choices = {}
            for req in event['requests']:
                u = req['uav']; key = f"{u}:{req['pointer']}"
                if key in plan['truncations']:
                    choices[u] = dict(truncate=True, T=ex.state.N, route=plan['truncations'][key])
                elif req['reason'] == 'deferral':
                    choices[u] = deepcopy(plan['deferrals'].get(f'{key}:{ex.state.n}', dict(T=ex.state.n, route='keep')))
                else:
                    route = plan['routes'][key]
                    r = req['routes'][route]
                    choices[u] = dict(T=r['earliest']+plan['offsets'][key], route=route)
            decision = dict(commitments=choices)
        ex.apply(decision)
    return ex.result()


def el_moves(instance, plan, audit, rng):
    padded = deepcopy(plan)
    for k in range(len(instance['tasks'])):
        padded['locations'].setdefault(str(k), 0)
    for u, visits in enumerate(instance['visits']):
        for j in range(len(visits)):
            padded['offsets'].setdefault(f'{u}:{j}', 0)
            padded['routes'].setdefault(f'{u}:{j}', 0)
    for move, candidate in moves(instance, padded, audit, rng):
        field, key = move['field'], move['key']
        if field != 'layers' and key not in plan[field]:
            continue
        actual = deepcopy(plan)
        if field == 'layers':
            actual[field] = candidate[field]
        else:
            actual[field][key] = candidate[field][key]
        yield move, actual


def search(instance, budget=600., *, search_seed=0, lsc_evaluations=64,
           lsc_cpu_s=240., el_attempts=10000, execution_layer='evaluation'):
    if not 1 <= lsc_evaluations <= 500 or el_attempts < 0 or lsc_cpu_s <= 0:
        raise ValueError('invalid search work limit')
    if execution_layer not in ('evaluation', 'audit'):
        raise ValueError('execution_layer must be evaluation or audit')
    minimum = math.ceil(.85*len(instance['tasks'])-1e-9)
    tic, wall = time.process_time(), time.perf_counter()
    stages, starts, trace, lsc_trace, incumbent = {}, [], [], [], None
    completed = rejected = attempts = 0
    active_stage = None; stop = 'work_limit_or_neighbourhood_exhausted'
    tb = None

    @contextmanager
    def stage(name):
        nonlocal active_stage
        active_stage = name; start = time.process_time()
        stages[name] = dict(cpu_s=0., status='running')
        try:
            yield
            stages[name]['status'] = 'complete'
        except (CpuLimit, SystemError) as exc:
            if is_cpu_limit(exc):
                stages[name]['status'] = 'cpu_budget'
            raise
        finally:
            stages[name]['cpu_s'] = time.process_time()-start

    def accept(run, name, stage_name, commitment=None, variant='earliest_feasible', move=None):
        nonlocal incumbent, completed
        if run.get('certified_service') is not True:
            raise AssertionError('certified service must be enabled')
        if execution_layer == 'audit' and (run.get('execution_layer') != 'audit' or
                run.get('terminal_replay') is not True or run.get('audit_source') != 'independent_replay'):
            raise AssertionError('audit certification requires audit execution and independent terminal replay')
        completed += 1
        key = list(objective(run['audit'], minimum))
        best = dict(method=name, start=name, stage=stage_name, objective=key,
                    result=run, audit_feasible=feasible(run['audit'], minimum),
                    energies_upper_J=certified_energies(run['audit']),
                    plan=realized_plan(run, instance), commitment=commitment, variant=variant)
        improved = incumbent is None or tuple(key) < tuple(incumbent['objective'])
        trace.append(dict(evaluation=completed, stage=stage_name, start=name, move=move,
                          objective=key, accepted=improved, cpu_s=time.process_time()-tic))
        if improved:
            incumbent = best
        return best

    def run_start(name, function, commitment=None, variant='earliest_feasible'):
        nonlocal rejected
        try:
            best = accept(function(), name, active_stage, commitment, variant)
            starts.append(dict(start=name, objective=best['objective'], stage=active_stage))
            return best
        except (Uncertified, ValueError) as exc:
            rejected += 1
            starts.append(dict(start=name, stage=active_stage, error=str(exc)))
            return None

    try:
        with deadline(budget):
            with stage('el_starts'):
                lib = PrimitiveLibrary(instance); planner = MotionPlanner(lib, max_alternatives=3)
                for method in METHODS:
                    run_start(method, lambda m=method: el_rollout(instance, m, library=lib, planner=planner,
                                                               execution_layer=execution_layer))
            with stage('dp_build'):
                tb = CommitmentTables(instance, executor=EventExecutor(instance, library=lib, planner=planner,
                                                                      execution_layer=execution_layer))
                dp = RelaxedDP(tb); reference = dp.commitment()
            with stage('dp_starts'):
                for variant in VARIANTS:
                    run_start('DP-Commit:'+variant, lambda v=variant: rollout(instance, reference, tb, v,
                        execution_layer=execution_layer), reference, variant)
            with stage('eks_16'):
                combinations = dp.reference_combinations(16)
                stages['eks_16']['constructed'] = len(combinations)
                eks_best = None
                for i, (_, commitment) in enumerate(combinations):
                    best = run_start(f'EKS-16:{i:02d}', lambda c=commitment: rollout(instance, c, tb,
                        execution_layer=execution_layer), commitment)
                    if best is not None and (eks_best is None or tuple(best['objective']) < tuple(eks_best['objective'])):
                        eks_best = best
                stages['eks_16']['best'] = None if eks_best is None else eks_best['start']
            with stage('ls_c'):
                ls_start = time.process_time(); ls_best = None
                def callback(entry, commitment, run):
                    nonlocal ls_best
                    lsc_trace.append(dict(entry))
                    stages['ls_c']['completed_evaluations'] = entry['evaluation']
                    if entry['accepted']:
                        ls_best = accept(run, 'LS-C', 'ls_c', commitment, move=entry['move'])
                    if time.process_time()-ls_start >= lsc_cpu_s:
                        raise CommitmentLimit()
                try:
                    found = local_search(instance, tb, reference, budget=lsc_evaluations,
                                         seed=search_seed, callback=callback, execution_layer=execution_layer)
                    stages['ls_c']['stop'] = found['stop']
                except CommitmentLimit:
                    stages['ls_c']['stop'] = 'stage_cpu_limit_at_completed_evaluation'
                if ls_best is not None:
                    starts.append(dict(start='LS-C', stage='ls_c', objective=ls_best['objective']))
            with stage('ls_el'):
                if incumbent is not None:
                    replayed = plan_rollout(instance, incumbent['plan'], tb, incumbent['commitment'], incumbent['variant'],
                                            execution_layer=execution_layer)
                    if replayed['logs'] != incumbent['result']['logs']:
                        different = next(i for i, (a, b) in enumerate(zip(replayed['logs'], incumbent['result']['logs'])) if a != b)
                        fields = [k for k in replayed['logs'][different] if replayed['logs'][different][k] != incumbent['result']['logs'][different].get(k)]
                        raise AssertionError(f"VCP-to-EL bridge changed {incumbent['start']} at slot {different}, fields {fields}")
                    stages['ls_el']['start'] = incumbent['start']
                    stages['ls_el']['start_objective'] = incumbent['objective']
                    stages['ls_el']['bridge_replays'] = 1
                    seen = {digest(incumbent['plan'])}; rng = np.random.default_rng(search_seed)
                    while attempts < el_attempts:
                        changed = False; current = incumbent
                        for move, candidate in el_moves(instance, current['plan'], current['result']['audit'], rng):
                            signature = digest(candidate)
                            if signature in seen:
                                continue
                            seen.add(signature); attempts += 1
                            try:
                                run = plan_rollout(instance, candidate, tb, current['commitment'], current['variant'],
                                                   execution_layer=execution_layer)
                                accept(run, current['start'], 'ls_el', current['commitment'], current['variant'], move)
                                changed = incumbent is not current
                            except (Uncertified, ValueError, IndexError, KeyError) as exc:
                                rejected += 1
                                trace.append(dict(stage='ls_el', move=move, rejected=str(exc), cpu_s=time.process_time()-tic))
                            if changed or attempts >= el_attempts:
                                break
                        if not changed:
                            break
    except (CpuLimit, SystemError) as exc:
        if not is_cpu_limit(exc):
            raise
        stop = 'cpu_budget'
    cpu_s = time.process_time()-tic
    if incumbent is not None:
        best_key = min((tuple(s['objective']) for s in starts if 'objective' in s), default=tuple(incumbent['objective']))
        assert tuple(incumbent['objective']) <= best_key
    return dict(version=VERSION, status='witness' if incumbent and incumbent['audit_feasible'] else 'none',
                incumbent=incumbent, starts=starts, trace=trace, lsc_trace=lsc_trace, stages=stages,
                c_min=minimum, budget_s=budget, cpu_s=cpu_s, wall_s=time.perf_counter()-wall,
                stop=stop, completed_evaluations=completed+len(lsc_trace)-sum(t['accepted'] for t in lsc_trace),
                bridge_replays=stages.get('ls_el', {}).get('bridge_replays', 0), rejected_evaluations=rejected,
                el_attempts=attempts, lsc_evaluation_limit=lsc_evaluations, lsc_cpu_limit=lsc_cpu_s,
                el_attempt_limit=el_attempts)


def anchor(instance, result, margin=.1):
    final = anchor_instance(instance, result, margin)
    record = final['certification']; record['planner_version'] = VERSION
    best = result['incumbent']
    record.update(start=None if best is None else best['start'], stage=None if best is None else best['stage'],
                  stages=result['stages'])
    return final


def metrics(run, instance):
    audit = run['audit']; p = instance['params']; tasks = audit['tasks']
    row = summary_audit(audit)
    row['hard_nonpass'] = hard_nonpass(audit)
    row.update(readiness_counts(audit))
    row['unknown_tasks'] = [t['k'] for t in tasks if t['result_ready'] == 'unknown']
    priorities = {d['slot']: d['choices']['priorities'] for d in run['decisions'] if d['kind'] == 'priorities'}
    served = [set() for _ in range(p['M'])]
    busy = requests = contested = 0
    for log in run['logs']:
        n = log['n']; by_bs = [set() for _ in range(p['M'])]
        for k, task in enumerate(tasks):
            m = task['location']
            if m is None or m == 0:
                continue
            events = run['service_events'][k]
            release, end = events['g']['value'], events['ul']['value']
            if release is not None and release <= n and (end is None or n < end) and priorities[n][k][0] > 0:
                by_bs[m-1].add(task['u'])
            if any(v > 0 for v in log['sigma'][k]):
                served[m-1].add(task['u'])
        counts = [len(v) for v in by_bs]
        busy += sum(v > 0 for v in counts)
        requests += sum(counts)
        contested += any(v >= 2 for v in counts)
    row.update(slots=len(run['logs']), contested_slots=contested, busy_bs_slots=busy,
               uplink_requests=requests, bs_serves_multiple_uavs=any(len(v) >= 2 for v in served),
               served_uavs_by_bs=[sorted(v) for v in served])
    return row


def serializable(value):
    if hasattr(value, '__dataclass_fields__'):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
