from fractions import Fraction
from functools import lru_cache
import math
import numpy as np

from lawn_mec.env.interval import enclose_integral, iv
from lawn_mec.env.model import rate_interval
from lawn_mec.env.queue import Event
from .state import clone_task


@lru_cache(maxsize=32)
def _idle_capacity(delta, subdivisions):
    return enclose_integral(lambda t: iv(0), delta, subdivisions)


def audit_radio_capacity(state, slot, k, stage):
    if hasattr(state, 'physics'):
        from .physics import audit_radio_capacity as cached_capacity
        return cached_capacity(state, slot, k, stage)
    p = state.p
    m, u = slot['locations'][k], state.tasks[k]['u']
    if m in (None, 0):
        return iv(0)
    direction = stage // 2
    sigma = slot['sigma'][k][direction]
    power = slot['power'][k] if stage == 0 else p['P_dl']*sigma
    subdivisions = state.cfg['integral_subdivisions']
    if not sigma or power == 0:
        return _idle_capacity(p['delta'], subdivisions)
    args = [np.asarray(slot[key])[u] for key in ('q', 'v', 'accel')]
    args += [state.instance['bs'][m-1], state.instance['buildings'], power, sigma, p]
    return enclose_integral(lambda t: rate_interval(*args, t), p['delta'], subdivisions)


def _float_ceiling(value):
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


class CertifiedService:
    def __init__(self, count):
        self.totals = [[Fraction(0) for _ in range(3)] for _ in range(count)]
        self.completions = [[None]*3 for _ in range(count)]

    def clone(self):
        other = CertifiedService(0)
        other.totals = [row.copy() for row in self.totals]
        other.completions = [row.copy() for row in self.completions]
        return other

    def _release(self, k, st, stage):
        return (st.events['g'].value if stage == 0 or st.location == 0
                else self.completions[k][stage-1])

    def task_view(self, k, task, st, n):
        out = clone_task(st)
        out.W[:] = 0
        for stage, name in enumerate(('ul', 'c', 'dl')):
            end = self.completions[k][stage]
            out.events[name] = Event(end is not None, end)
        end = self.completions[k][1 if st.location == 0 else 2]
        out.events['r'] = Event(end is not None, end)
        out.injected = set()
        if st.location is not None:
            for stage in ((1,) if st.location == 0 else (0, 1, 2)):
                release = self._release(k, st, stage)
                if release is not None and release <= n:
                    out.injected.add('g' if stage == 0 or st.location == 0 else ('ul', 'c')[stage-1])
                    if self.completions[k][stage] is None:
                        work = task[('D', 'C', 'O')[stage]]
                        out.W[stage] = _float_ceiling(Fraction(work)-self.totals[k][stage])
        return out

    def advance(self, state, slot):
        n = slot['n']
        for k, (task, st) in enumerate(zip(state.tasks, state.states)):
            if st.location is None:
                continue
            for stage in ((1,) if st.location == 0 else (0, 1, 2)):
                release = self._release(k, st, stage)
                if self.completions[k][stage] is not None or release is None or release > n:
                    continue
                cap = (Fraction(state.p['delta'])*Fraction(slot['f'][k]) if stage == 1
                       else Fraction(audit_radio_capacity(state, slot, k, stage).lo))
                self.totals[k][stage] += cap
                if self.totals[k][stage] >= task[('D', 'C', 'O')[stage]]:
                    self.completions[k][stage] = n+1
