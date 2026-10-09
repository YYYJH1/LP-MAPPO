from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import signal
import time
import numpy as np
from .motion import MotionPlanner
from .primitives import PrimitiveLibrary, Uncertified
from .policies import METHODS, rollout, episode_plan
from .p0_loss import search_key

VERSION = 'ls-el-stage23-1'


def objective(audit, c_min):
    return search_key(audit, c_min)


def improves(candidate, incumbent):
    return incumbent is None or tuple(candidate) < tuple(incumbent)


def feasible(audit, c_min=None):
    c_min = math.ceil(.85*len(audit['tasks'])-1e-9) if c_min is None else c_min
    return objective(audit, c_min)[:2] == (0, 0)


def certified_energies(audit):
    return [math.nextafter(hi+2e-12*(hi-lo), math.inf)
            for lo, hi in (u['energy_interval'] for u in audit['uavs'])]


class BudgetExpired(Exception):
    pass


@contextmanager
def cpu_deadline(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite CPU budget required')
    old_handler, old_timer = signal.getsignal(signal.SIGPROF), signal.getitimer(signal.ITIMER_PROF)
    if old_timer != (0., 0.):
        raise RuntimeError('nested profiling timers are unsupported')
    def expire(signum, frame):
        raise BudgetExpired()
    signal.signal(signal.SIGPROF, expire)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old_handler)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def moves(instance, plan, audit, rng):
    tasks = sorted(audit['tasks'], key=lambda t: (t['completed'], t['k']))
    candidates = []
    for task in tasks:
        k, u = task['k'], task['u']
        loc = plan['locations'][str(k)]
        for m in range(instance['params']['M']+1):
            if m != loc:
                candidates.append(('locations', str(k), m))
        for j, visit in enumerate(instance['visits'][u]):
            if visit[0] != k:
                continue
            key = f'{u}:{j}'; offset = plan['offsets'][key]
            if visit[1] == 'a' and task['n_r'] is not None and task['n_a'] is not None and task['n_r'] > task['n_a']:
                candidates.append(('offsets', key, offset+task['n_r']-task['n_a']))
            for shift in (1, 4, 12, -1, -4, -12, 24):
                if offset+shift >= 0:
                    candidates.append(('offsets', key, offset+shift))
            for route in range(4):
                if route != plan['routes'][key]:
                    candidates.append(('routes', key, route))
    layer_moves = []
    for u in range(instance['params']['U']):
        for ell in range(7):
            if ell != plan['layers'][u]:
                new = plan['layers'].copy()
                if ell in new:
                    w = new.index(ell); new[w] = new[u]
                new[u] = ell
                layer_moves.append(('layers', None, new))
    groups = [[c for c in candidates if c[0] == name] for name in ('locations', 'offsets', 'routes')]
    groups.append(layer_moves)
    ordered = []
    for i in range(max(map(len, groups), default=0)):
        for group in groups:
            if i < len(group):
                ordered.append(group[i])
    seen = set()
    for field, key, value in ordered:
        candidate = deepcopy(plan)
        if field == 'layers':
            candidate[field] = value
        else:
            candidate[field][key] = value
        signature = digest(candidate)
        if signature not in seen:
            seen.add(signature)
            yield dict(field=field, key=key, value=value), candidate


def search(instance, budget=300., *, c_min=None, search_seed=0,
           execution_layer='evaluation'):
    if execution_layer not in ('evaluation', 'audit'):
        raise ValueError('execution_layer must be evaluation or audit')
    c_min = math.ceil(.85*len(instance['tasks'])-1e-9) if c_min is None else int(c_min)
    started, wall = time.process_time(), time.perf_counter()
    incumbent, trace, evaluated = None, [], set()
    completed, rejected, attempts = 0, 0, 0
    stop = 'neighbourhood_exhausted'; rng = np.random.default_rng(search_seed)
    def evaluate(method, plan, move):
        nonlocal incumbent, completed, rejected, attempts
        attempts += 1
        try:
            result = rollout(instance, method, plan, library=lib, planner=planner,
                             execution_layer=execution_layer)
        except (Uncertified, ValueError) as exc:
            rejected += 1
            trace.append(dict(cpu_s=time.process_time()-started, move=move, rejected=str(exc)))
            return False
        completed += 1
        key = objective(result['audit'], c_min)
        accepted = improves(key, None if incumbent is None else incumbent['objective'])
        trace.append(dict(cpu_s=time.process_time()-started, move=move, objective=list(key), accepted=accepted))
        if accepted:
            realized = episode_plan(result)
            incumbent = dict(method=method, plan=realized, objective=list(key), result=result,
                             audit_feasible=feasible(result['audit'], c_min),
                             energies_upper_J=certified_energies(result['audit']))
            if incumbent['audit_feasible'] and (key[0] or key[1]):
                raise AssertionError('unreplayed feasible incumbent')
        return accepted
    try:
        with cpu_deadline(budget):
            lib = PrimitiveLibrary(instance)
            planner = MotionPlanner(lib, max_alternatives=3)
            for method in METHODS:
                evaluate(method, None, dict(start=method))
            if incumbent is not None:
                while True:
                    changed = False
                    current = incumbent
                    for move, candidate in moves(instance, current['plan'], current['result']['audit'], rng):
                        signature = digest(candidate)
                        if signature in evaluated:
                            continue
                        evaluated.add(signature)
                        if evaluate(current['method'], candidate, move):
                            changed = True
                            break
                    if not changed:
                        break
    except BudgetExpired:
        stop = 'cpu_budget'
    cpu_s = time.process_time()-started
    return dict(version=VERSION, status='witness' if incumbent and incumbent['audit_feasible'] else 'none',
                incumbent=incumbent, trace=trace, budget_s=budget, cpu_s=cpu_s,
                wall_s=time.perf_counter()-wall, stop=stop, c_min=c_min,
                completed_evaluations=completed, rejected_evaluations=rejected, attempts=attempts,
                evaluations_per_cpu_s=completed/cpu_s if cpu_s else 0.)
