from dataclasses import dataclass
import math
import numpy as np

_TINY = float(np.finfo(float).tiny)
_HUGE = 1e308
_INF = math.inf
_NINF = -math.inf
_nextafter = math.nextafter


def _rounded_numpy(cls, lo, hi):
    eps = 1e-12 * max(abs(lo), abs(hi), np.finfo(float).tiny)
    return cls(float(np.nextafter(lo-eps, -np.inf)), float(np.nextafter(hi+eps, np.inf)))


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi or math.isnan(self.lo) or math.isnan(self.hi):
            raise ValueError('invalid interval')

    @classmethod
    def point(cls, x):
        return cls(float(x), float(x))

    @classmethod
    def rounded(cls, lo, hi):
        if type(lo) is float and type(hi) is float:
            m = abs(lo); b = abs(hi)
            if b > m:
                m = b
            if m > _HUGE:
                return _rounded_numpy(cls, lo, hi)
            if m < _TINY:
                if hi != hi:
                    return _rounded_numpy(cls, lo, hi)
                m = _TINY
            eps = 1e-12*m
            return cls(_nextafter(lo-eps, _NINF), _nextafter(hi+eps, _INF))
        return _rounded_numpy(cls, lo, hi)

    def __add__(self, other):
        o = iv(other)
        return self.rounded(self.lo+o.lo, self.hi+o.hi)
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -iv(other)

    def __rsub__(self, other):
        return iv(other) + -self

    def __mul__(self, other):
        o = iv(other)
        xs = [self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi]
        return self.rounded(min(xs), max(xs))
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = iv(other)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError('interval contains zero')
        return self * self.rounded(1/o.hi, 1/o.lo)

    def __rtruediv__(self, other):
        return iv(other)/self

    def __pow__(self, power):
        if power == 0:
            return iv(1)
        if int(power) == power and power > 0:
            power = int(power)
            xs = [self.lo**power, self.hi**power]
            lower = 0 if power % 2 == 0 and self.lo <= 0 <= self.hi else min(xs)
            return self.rounded(lower, max(xs))
        if self.lo <= 0:
            raise ValueError('noninteger power needs a strictly positive interval')
        xs = [self.lo**power, self.hi**power]
        return self.rounded(min(xs), max(xs))

    def sqrt(self):
        if self.hi < 0:
            raise ValueError('negative square root')
        r = self.rounded(math.sqrt(max(0, self.lo)), math.sqrt(max(0, self.hi)))
        return Interval(max(0, r.lo), r.hi)

    def log2(self):
        if self.lo <= 0:
            raise ValueError('nonpositive log')
        return self.rounded(math.log2(self.lo), math.log2(self.hi))

    def as_list(self):
        return [self.lo, self.hi]


def iv(x):
    return x if isinstance(x, Interval) else Interval.point(x)


def norm(xs):
    return sum((iv(x)**2 for x in xs), iv(0)).sqrt()


def enclose_integral(fn, delta, subdivisions):
    total = iv(0)
    for lo, hi in zip(np.linspace(0, delta, subdivisions+1)[:-1],
                      np.linspace(0, delta, subdivisions+1)[1:]):
        total += fn(Interval(float(lo), float(hi))) * (hi-lo)
    return total


def certify_lower(fn, delta, threshold, budget):
    pending, count, low, high = [(0., delta)], 0, float('inf'), float('inf')
    unresolved = False
    while pending:
        lo, hi = pending.pop()
        r = fn(Interval(lo, hi))
        count += 1
        low, high = min(low, r.lo), min(high, r.hi)
        if r.hi < threshold:
            return {'status': 'fail', 'residual': threshold-r.hi, 'boxes': count}
        if r.lo >= threshold:
            continue
        mid = (lo+hi)/2
        point = fn(Interval.point(mid))
        if point.hi < threshold:
            return {'status': 'fail', 'residual': threshold-point.hi, 'boxes': count}
        if count+len(pending)+2 > budget or mid in (lo, hi):
            unresolved = True
        else:
            pending.extend([(lo, mid), (mid, hi)])
    return {'status': 'unknown' if unresolved else 'pass',
            'residual': threshold-low if unresolved else 0., 'boxes': count}
