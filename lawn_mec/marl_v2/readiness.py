from dataclasses import dataclass
from fractions import Fraction
import math

from lawn_mec.env.model import slot_capacity
from lawn_mec.exec_v2.policies import forecast_motion

REQUEST_VALUES = 7
CANDIDATE_COLUMNS = (13, 14)


@dataclass(frozen=True)
class Readiness:
    certified: bool
    lo: int
    hi: int
    earliest: int
    limit: int
    salvageable: bool
    values: tuple

    def candidate(self, T, N):
        return [float(T >= self.hi), (T-self.hi)/N]


def serial_completion(executor, k, loc, view, W, ahead, trajectory=None):
    s = executor.state; task, p = s.tasks[k], s.p
    release = s.n
    if loc == 0:
        if s.layer == 'audit':
            capacity = Fraction(p['delta'])*Fraction(p['F_local'])
            ahead = max(Fraction(0), ahead-(release-s.n)*capacity)
            return min(s.N+1, release+math.ceil((ahead+W[1])/capacity))
        ahead = max(0., ahead-(release-s.n)*p['delta']*p['F_local'])
        return min(s.N+1, release+math.ceil((ahead+W[1])/(p['delta']*p['F_local'])))
    if s.layer == 'audit':
        ahead = max(Fraction(0), ahead-sum((Fraction(p['delta'])*Fraction(
            p['G_total']-s.instance['load'][loc-1][n])
            for n in range(s.n, min(release, s.N))), Fraction(0)))
    else:
        ahead = max(0., ahead-sum(p['delta']*(p['G_total']-s.instance['load'][loc-1][n])
                                 for n in range(s.n, min(release, s.N))))
    trajectory = trajectory if trajectory is not None else forecast_motion(executor, task['u'])
    if W[2] > 0 or view.events['c'].available:
        stage = 2
    elif W[1] > 0 or view.events['ul'].available:
        stage = 1
    else:
        stage = 0
    for offset, (q, v, a) in enumerate(trajectory):
        n = s.n+offset
        if n < release:
            continue
        if stage == 1:
            cap = (Fraction(p['delta'])*Fraction(p['G_total']-s.instance['load'][loc-1][n])
                   if s.layer == 'audit' else p['delta']*(p['G_total']-s.instance['load'][loc-1][n]))
            used = min(ahead, cap); ahead -= used; cap -= used
        else:
            if s.layer == 'audit':
                cap = Fraction(s.physics.radio(q, v, a, loc-1,
                                               p['P_ul'] if stage == 0 else p['P_dl'], 1.).lo)
            else:
                cap = slot_capacity(q, v, a, s.instance['bs'][loc-1], s.instance['buildings'],
                                    p['P_ul'] if stage == 0 else p['P_dl'], 1., p, 'guard', s.cfg)
        W[stage] -= cap
        if W[stage] <= 0:
            if stage == 2:
                return n+1
            stage += 1; W[stage] = task['C'] if stage == 1 else task['O']
            if s.layer == 'audit':
                W[stage] = Fraction(W[stage])
    return s.N+1


def readiness_bounds(executor, k, views, own):
    s = executor.state; task, st, p = s.tasks[k], s.states[k], s.p
    view = views[k]
    if view.events['r'].available:
        return True, view.events['r'].value, view.events['r'].value
    loc = view.location
    W = view.W.copy()
    if not st.events['g'].available:
        W[1 if loc == 0 else 0] = task['C'] if loc == 0 else task['D']
    weight = {j: 1./(max(0., s.tasks[j]['d']-s.n*p['delta'])+1.) for j in own}
    key = lambda j: (-weight[j]/s.tasks[j]['C'], j)
    ahead_own = [j for j in own if j != k and views[j].location == loc and key(j) < key(k)]
    audit = s.layer == 'audit'
    if audit:
        W = [Fraction(float(x)) for x in W]
        lo_ahead = sum((Fraction(float(views[j].W[1])) for j in ahead_own), Fraction(0))
    else:
        lo_ahead = sum(views[j].W[1] for j in ahead_own)
    others = Fraction(0)
    if loc != 0:
        broadcast = sum((Fraction(float(v.W[1])) for v in views if v.location == loc), Fraction(0))
        others = broadcast-sum((Fraction(float(views[j].W[1])) for j in own if views[j].location == loc),
                               Fraction(0))
    trajectory = None if loc == 0 else forecast_motion(executor, task['u'])
    lo = serial_completion(executor, k, loc, view, list(W) if audit else W.copy(), lo_ahead, trajectory)
    if others <= 0:
        return False, lo, lo
    hi_ahead = lo_ahead+others if audit else lo_ahead+float(others)
    return False, lo, serial_completion(executor, k, loc, view, list(W) if audit else W.copy(), hi_ahead,
                                        trajectory)


def timing_slots(env, request):
    lo = request['earliest']; hi = request['latest']
    if request['keep']:
        lo = min(lo, request['keep_T']); hi = max(hi, request['keep_T'])
    legal = env._legal_times(request, lo, hi)
    return [lo+int(i) for i in legal.nonzero()[0]]


def request_readiness(env, request):
    visit = request.get('visit')
    if visit is None or visit[1] != 'a':
        return None
    k = visit[0]; ex = env.executor; s = ex.state; u = request['uav']
    task = s.tasks[k]
    if task['u'] != u:
        raise ValueError('a visit request for another UAV task')
    views = env.task_states()
    if views[k].location is None:
        return None
    times = timing_slots(env, request)
    if not times:
        return None
    certified, lo, hi = readiness_bounds(ex, k, views, env._own_tasks[u])
    N, n = env.N, s.n
    ns = views[k].events['s'].value
    deadline = math.floor(task['d']/s.p['delta'])-task['L_a']
    fresh = ns+math.floor(task['H']/s.p['delta']) if ns is not None else -1
    earliest = times[0]; limit = min(times[-1], deadline, fresh)
    start = max(earliest, hi)
    salvageable = any(start <= T <= limit for T in times)
    values = (1., float(certified), (hi-n)/N, (lo-n)/N, float(hi > N), float(salvageable),
              min(1., max(-1., (limit-start)/N)))
    return Readiness(certified, lo, hi, earliest, limit, salvageable, values)
