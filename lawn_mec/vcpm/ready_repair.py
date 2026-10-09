import math
import numpy as np


def init_counts(env):
    env.pace_counts.update(repair_triggers=0, repair_wait_slots=0, repair_abandons=0)
    env._repair_wait_request = None


def suffix_limit(env, request):
    ex = env.executor; state = ex.state; u = request['uav']
    visits = state.instance['visits'][u]; pointer = request['pointer']
    ell = ex.layers[u]; travel = ex.planner.minimum_times[ell]
    views = env.task_states(); delta = state.p['delta']
    end, dest = ex.planner.landing_start, ex.final_nodes[u]
    latest = end
    for k, phase in reversed(visits[pointer:]):
        task = state.tasks[k]; node = task['z_'+phase]
        latest = end-int(travel[node, dest])-task['L_'+phase]
        if phase == 'a':
            latest = min(latest, math.floor(task['d']/delta)-task['L_a'])
            ns = views[k].events['s'].value
            if ns is not None:
                latest = min(latest, ns+math.floor(task['H']/delta))
        end, dest = latest, node
    return min(int(request['latest']), int(latest))


def repair_support(env, times, mask, request):
    env._repair_wait_request = None
    r = env.readiness(request)
    if r is None or r.certified:
        return None
    env.pace_counts['repair_triggers'] += 1
    n = env.executor.state.n
    limit = min(r.limit, suffix_limit(env, request))
    earliest = max(r.earliest, n)
    target = max(earliest, r.hi)
    k, phase = request['visit']; c = env.plan.tasks[k]
    lo, band_hi = earliest, limit
    if c.g == 'S' and c.a is not None:
        from .features import interval
        band_lo, band_hi = interval(env.tables, k, phase, c.a)
        lo = max(lo, band_lo)
    support = mask & (times >= max(lo, target)) & (times <= min(limit, band_hi))
    if not support.any():
        support = mask & (times >= target) & (times <= limit)
    if support.any():
        env._repair_wait_request = (dict(request), earliest)
        return support, True, False, True
    env.pace_counts['repair_abandons'] += 1
    legal = np.flatnonzero(mask)
    support = mask & (times == times[legal[np.argmin(times[legal])]])
    return support, True, False, True


def record_wait(env, head, request, candidates, action):
    pending = env._repair_wait_request
    if head == 'timing' and pending is not None and pending[0] == request:
        selected = int(round(float(candidates[action, 0])*env.N))
        env.pace_counts['repair_wait_slots'] += max(0, selected-pending[1])
        env._repair_wait_request = None
