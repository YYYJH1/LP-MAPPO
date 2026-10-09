from dataclasses import dataclass
import numpy as np

_FACE_NORMALS = tuple((np.eye(3)[i], -np.eye(3)[i]) for i in range(3))
for _pair in _FACE_NORMALS:
    for _normal in _pair: _normal.flags.writeable = False

@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        self.lo, self.hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        if self.lo.shape != (3,) or self.hi.shape != (3,):
            raise ValueError('boxes require three coordinates')

    @classmethod
    def from_dict(cls, d):
        return cls(d['lo'], d['hi'])

    def as_dict(self):
        return {'lo': self.lo.tolist(), 'hi': self.hi.tolist()}

    @property
    def nonempty(self):
        return bool(np.all(self.lo <= self.hi))

    @property
    def center(self):
        return (self.lo + self.hi)/2

    def contains(self, q, tol=0):
        q = np.asarray(q)
        if q.shape == (3,):
            return bool(q[0] >= self.lo[0]-tol and q[1] >= self.lo[1]-tol
                        and q[2] >= self.lo[2]-tol and q[0] <= self.hi[0]+tol
                        and q[1] <= self.hi[1]+tol and q[2] <= self.hi[2]+tol)
        return bool(np.all(q >= self.lo-tol) and np.all(q <= self.hi+tol))

    def erode(self, s):
        lo, hi = self.lo+s, self.hi-s
        if lo.shape != (3,) or hi.shape != (3,): return Box(lo, hi)
        box = object.__new__(Box)
        box.lo, box.hi = lo, hi
        return box

    def intersect(self, other):
        return Box(np.maximum(self.lo, other.lo), np.minimum(self.hi, other.hi))

    def distance(self, other):
        return float(np.linalg.norm(np.maximum(0, np.maximum(self.lo-other.hi, other.lo-self.hi))))

    def point_distance(self, q):
        return self.distance(Box(q, q))

    def faces(self):
        return [(h, b) for i in range(3) for h, b in
                [(np.eye(3)[i], self.hi[i]), (-np.eye(3)[i], -self.lo[i])]]


def segment_box(p, q, box, open_segment=True):
    p, q = np.asarray(p, float), np.asarray(q, float)
    t0, t1 = 0., 1.
    for i, d in enumerate(q-p):
        if d == 0:
            if p[i] < box.lo[i] or p[i] > box.hi[i]:
                return False
        else:
            a, b = sorted(((box.lo[i]-p[i])/d, (box.hi[i]-p[i])/d))
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
    return bool(t0 <= t1 and (not open_segment or (t1 > 0 and t0 < 1)))


def trajectory_box_residual(q, v, a, box, delta):
    worst = -float('inf')
    for h, b in ((h, bound) for i, normals in enumerate(_FACE_NORMALS)
                 for h, bound in ((normals[0], box.hi[i]), (normals[1], -box.lo[i]))):
        c, d, x = h@q-b, h@v, h@a
        ts = [0., delta]
        if x != 0 and 0 < -d/x < delta:
            ts.append(-d/x)
        worst = max(worst, *(c+d*t+x*t*t/2 for t in ts))
    return float(worst)


def face_accel_bound(c, d, delta):
    if c > 0 or (c == 0 and d > 0):
        return -float('inf')
    ts = [delta]
    if d != 0 and 0 < -2*c/d < delta:
        ts.append(-2*c/d)
    values = [-2*c/t**2-2*d/t for t in ts]
    if c == 0 and d == 0:
        values.append(0.)
    return min(values)
