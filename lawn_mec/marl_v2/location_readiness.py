from fractions import Fraction
import math

from lawn_mec.env.model import slot_capacity
from lawn_mec.env.queue import inject
from lawn_mec.exec_v2.policies import forecast_motion
from lawn_mec.exec_v2.state import clone_task

COLUMNS = tuple(range(9, 15))
VALUES = len(COLUMNS)


class EventCache:
    def __init__(self, env):
        self.n = env.executor.state.n
        self.views = env.task_states()
        self.trajectories = {}
        self.capacities = {}

    def trajectory(self, executor, u):
        if u not in self.trajectories:
            self.trajectories[u] = forecast_motion(executor, u)
        return self.trajectories[u]

    def radio(self, executor, u, m, stage, offset):
        key = (u, m, stage, offset)
        if key not in self.capacities:
            s = executor.state; p = s.p
            q, v, a = self.trajectories[u][offset]
            power = p['P_ul'] if stage == 0 else p['P_dl']
            if s.layer == 'audit':
                self.capacities[key] = Fraction(s.physics.radio(q, v, a, m-1, power, 1.).lo)
            else:
                self.capacities[key] = slot_capacity(q, v, a, s.instance['bs'][m-1], s.instance['buildings'],
                                                     power, 1., p, 'guard', s.cfg)
        return self.capacities[key]


def prospective_view(executor, k, m):
    s = executor.state
    view = clone_task(s.states[k])
    view.location = m
    inject(s.tasks[k], view, s.n)
    if executor.service is not None:
        view = executor.service.task_view(k, s.tasks[k], view, s.n)
    return view


def initial_work(executor, k, m, view):
    s = executor.state; task = s.tasks[k]
    W = view.W.copy()
    if not s.states[k].events['g'].available:
        W[1 if m == 0 else 0] = task['C'] if m == 0 else task['D']
    if s.layer == 'audit':
        W = [Fraction(float(x)) for x in W]
    return W


def serial_completion(executor, k, m, view, W, ahead, radio):
    s = executor.state; task, p = s.tasks[k], s.p
    release = s.n
    if m == 0:
        if s.layer == 'audit':
            capacity = Fraction(p['delta'])*Fraction(p['F_local'])
            ahead = max(Fraction(0), ahead-(release-s.n)*capacity)
            return min(s.N+1, release+math.ceil((ahead+W[1])/capacity))
        ahead = max(0., ahead-(release-s.n)*p['delta']*p['F_local'])
        return min(s.N+1, release+math.ceil((ahead+W[1])/(p['delta']*p['F_local'])))
    if s.layer == 'audit':
        ahead = max(Fraction(0), ahead-sum((Fraction(p['delta'])*Fraction(
            p['G_total']-s.instance['load'][m-1][n])
            for n in range(s.n, min(release, s.N))), Fraction(0)))
    else:
        ahead = max(0., ahead-sum(p['delta']*(p['G_total']-s.instance['load'][m-1][n])
                                 for n in range(s.n, min(release, s.N))))
    if W[2] > 0 or view.events['c'].available:
        stage = 2
    elif W[1] > 0 or view.events['ul'].available:
        stage = 1
    else:
        stage = 0
    for offset in range(s.N-s.n):
        n = s.n+offset
        if n < release:
            continue
        if stage == 1:
            cap = (Fraction(p['delta'])*Fraction(p['G_total']-s.instance['load'][m-1][n])
                   if s.layer == 'audit' else p['delta']*(p['G_total']-s.instance['load'][m-1][n]))
            used = min(ahead, cap); ahead -= used; cap -= used
        else:
            cap = radio(stage, offset)
        W[stage] -= cap
        if W[stage] <= 0:
            if stage == 2:
                return n+1
            stage += 1; W[stage] = task['C'] if stage == 1 else task['O']
            if s.layer == 'audit':
                W[stage] = Fraction(W[stage])
    return s.N+1


def readiness_bounds(executor, k, m, views, own, radio):
    s = executor.state; p = s.p
    audit = s.layer == 'audit'
    view = prospective_view(executor, k, m)
    weight = {j: 1./(max(0., s.tasks[j]['d']-s.n*p['delta'])+1.) for j in own}
    key = lambda j: (-weight[j]/s.tasks[j]['C'], j)
    ahead_own = [j for j in own if j != k and views[j].location == m and key(j) < key(k)]
    if audit:
        lo_ahead = sum((Fraction(float(views[j].W[1])) for j in ahead_own), Fraction(0))
    else:
        lo_ahead = sum(views[j].W[1] for j in ahead_own)
    others = Fraction(0)
    if m != 0:
        broadcast = sum((Fraction(float(v.W[1])) for v in views if v.location == m), Fraction(0))
        others = broadcast-sum((Fraction(float(views[j].W[1])) for j in own if views[j].location == m),
                               Fraction(0))
    lo = serial_completion(executor, k, m, view, initial_work(executor, k, m, view), lo_ahead, radio)
    hi = lo if others <= 0 else serial_completion(
        executor, k, m, view, initial_work(executor, k, m, view),
        lo_ahead+others if audit else lo_ahead+float(others), radio)
    dur = lo if lo_ahead == 0 else serial_completion(
        executor, k, m, view, initial_work(executor, k, m, view), Fraction(0) if audit else 0., radio)
    return lo, hi, dur


def location_values(env, k, locations, cache):
    ex = env.executor; s = ex.state; p = s.p
    n, N = s.n, env.N
    if cache.n != n:
        raise RuntimeError('F16 event cache is stale')
    task = s.tasks[k]; u = task['u']
    views, own = cache.views, env._own_tasks[u]
    if views[k].location is not None:
        raise ValueError('F16 location features need an undecided task')
    ns = views[k].events['s'].value
    deadline = math.floor(task['d']/p['delta'])-task['L_a']
    fresh = ns+math.floor(task['H']/p['delta']) if ns is not None else -1
    common = [task['C']/1e10, (deadline-n)/N, (fresh-n)/N]
    rows = []
    for m in locations:
        if m != 0:
            cache.trajectory(ex, u)
        radio = lambda stage, offset, m=m: cache.radio(ex, u, m, stage, offset)
        lo, hi, dur = readiness_bounds(ex, k, m, views, own, radio)
        rows.append([(lo-n)/N, (hi-n)/N, (dur-n)/N, *common])
    return rows
