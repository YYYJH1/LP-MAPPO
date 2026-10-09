import math
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from .geometry import Box, segment_box
from .interval import Interval, iv, norm


def small_norm(x):
    x = np.asarray(x)
    if x.dtype != np.dtype('float64'): return np.linalg.norm(x)
    return np.float64(math.sqrt(float(x @ x)))


def _batch_norm(x):
    return np.sqrt((x[..., None, :] @ x[..., :, None])[..., 0, 0])


@lru_cache(maxsize=32)
def _quadrature(nodes):
    return np.polynomial.legendre.leggauss(nodes)


def motion(q, v, a, tau):
    return np.asarray(q)+tau*np.asarray(v)+tau*tau*np.asarray(a)/2, np.asarray(v)+tau*np.asarray(a)


def gain(q, bs, buildings, p):
    los = not any(segment_box(q, bs, Box.from_dict(b)) for b in buildings)
    d = np.linalg.norm(np.asarray(q)-bs)
    if d <= 0:
        raise ValueError('BS must lie outside the flight region')
    return p['beta_L' if los else 'beta_N']*(d/p['d0'])**(-p['alpha_L' if los else 'alpha_N'])


def rate(q, bs, buildings, power, sigma, p):
    return sigma*p['bandwidth']*np.log2(1+power*gain(q, bs, buildings, p)/(p['N0']*p['bandwidth']))


def integrate(fn, delta, layer, cfg, points=()):
    if layer == 'guard':
        xs, ws = _quadrature(cfg['guard_nodes'])
        return float(sum(w*fn((x+1)*delta/2) for x, w in zip(xs, ws))*delta/2)
    return float(quad(fn, 0, delta, points=sorted(set(points)), epsabs=cfg['quad_epsabs'],
                      epsrel=cfg['quad_epsrel'], limit=max(100, len(points)+20))[0])


def _small_polyroots(c):
    if isinstance(c, np.ndarray) and c.dtype == np.float64 and c.ndim == 1:
        length = len(c)
        while length > 1 and c[length-1] == 0:
            length -= 1
        if length == 1:
            return ()
        if length == 2:
            return (-c[0]/c[1],)
    return np.polynomial.polynomial.polyroots(c)


def _small_polysquare(c):
    if not (isinstance(c, np.ndarray) and c.dtype == np.float64 and c.ndim == 1 and len(c)):
        return np.polynomial.polynomial.polymul(c, c)
    length = len(c)
    while length > 1 and c[length-1] == 0:
        length -= 1
    result = np.convolve(c[:length], c[:length])
    length = len(result)
    while length > 1 and result[length-1] == 0:
        length -= 1
    return result[:length]


def los_switches(q, v, a, bs, buildings, delta, cfg):
    polys = [np.array([q[i], v[i], a[i]/2]) for i in range(3)]
    roots = [0., delta]
    for bd in buildings:
        box = Box.from_dict(bd)
        for i in range(3):
            for face in [box.lo[i], box.hi[i]]:
                c = polys[i].copy(); c[0] -= face
                roots += [float(r.real) for r in _small_polyroots(c)
                          if abs(r.imag) < 1e-9 and 0 < r.real < delta]
                for j in range(i):
                    for face2 in [box.lo[j], box.hi[j]]:
                        pi, pj = polys[i].copy(), polys[j].copy()
                        pi[0] -= bs[i]; pj[0] -= bs[j]
                        c = (face-bs[i])*pj-(face2-bs[j])*pi
                        roots += [float(r.real) for r in _small_polyroots(c)
                                  if abs(r.imag) < 1e-9 and 0 < r.real < delta]
    roots = sorted(set(roots))
    def blocked(t):
        x, _ = motion(q, v, a, t)
        return any(segment_box(x, bs, Box.from_dict(b)) for b in buildings)
    switches = []
    mids = [(x+y)/2 for x, y in zip(roots[:-1], roots[1:])]
    for left, right in zip(mids[:-1], mids[1:]):
        state = blocked(left)
        if state != blocked(right):
            for _ in range(45):
                mid = (left+right)/2
                if blocked(mid) == state:
                    left = mid
                else:
                    right = mid
            switches.append((left+right)/2)
    return switches


@lru_cache(maxsize=4096)
def _slot_capacity_cached(qkey, vkey, akey, bskey, geometry, power, sigma,
                          parameters, layer, settings, functions):
    q, v, a, bs = (np.frombuffer(x[2], dtype=x[0]).reshape(x[1])
                   for x in (qkey, vkey, akey, bskey))
    buildings = [dict(lo=lo, hi=hi) for lo, hi in geometry]
    return _slot_capacity_value(q, v, a, bs, buildings, power, sigma,
                                dict(parameters), layer, dict(settings))


def slot_capacity(q, v, a, bs, buildings, power, sigma, p, layer, cfg):
    if not sigma or power == 0: return 0.
    return _slot_capacity_cached(*(array_signature(x) for x in (q, v, a, bs)),
        tuple((tuple(b['lo']), tuple(b['hi'])) for b in buildings), power, sigma,
        tuple((key, p[key]) for key in ('delta', 'beta_L', 'beta_N', 'd0',
                                       'alpha_L', 'alpha_N', 'bandwidth', 'N0')),
        layer, tuple((key, cfg.get(key)) for key in ('guard_nodes', 'quad_epsabs', 'quad_epsrel')),
        (integrate, los_switches, rate, gain, segment_box, motion))


def _slot_capacity_value(q, v, a, bs, buildings, power, sigma, p, layer, cfg):
    if not sigma or power == 0:
        return 0.
    points = los_switches(q, v, a, bs, buildings, p['delta'], cfg) if layer != 'guard' else []
    return integrate(lambda t: rate(motion(q, v, a, t)[0], bs, buildings, power, sigma, p),
                     p['delta'], layer, cfg, points)


def frequency(f, owners, locations, u):
    return float(sum(x for x, owner, m in zip(f, owners, locations) if owner == u and m == 0))


def thrust(v, a, p):
    v, a = np.asarray(v), np.asarray(a)
    if v.ndim == 1 and a.ndim == 1:
        return small_norm(p['weight']/p['g0']*a+[0, 0, p['weight']]
                          +p['rho']*p['drag_area']/2*small_norm(v)*v)
    if v.dtype != np.dtype('float64') or a.dtype != np.dtype('float64'):
        vv, aa = np.broadcast_arrays(v, a)
        return np.array([thrust(x, y, p) for x, y in zip(vv.reshape(-1, 3), aa.reshape(-1, 3))]).reshape(vv.shape[:-1])
    return _batch_norm(p['weight']/p['g0']*a+[0, 0, p['weight']]
                       +p['rho']*p['drag_area']/2*_batch_norm(v)[..., None]*v)


def shaft_power(v, a, p):
    v, a = np.asarray(v), np.asarray(a)
    if (_interval_numba and v.dtype == np.float64 and a.dtype == np.float64
            and v.shape == (3,) and a.shape == (3,)):
        valid, value = _power_scalar(np.ascontiguousarray(v), np.ascontiguousarray(a),
                                     _iv_constants(p), _INTERVAL_POWERS)
        if valid:
            return float(value)
    if (_interval_numba and v.dtype == np.float64 and a.dtype == np.float64
            and v.shape[-1:] == (3,) and a.shape[-1:] == (3,)
            and np.all(np.isfinite(v)) and np.all(np.isfinite(a))):
        vv, aa = np.broadcast_arrays(v, a)
        flat_v, flat_a = vv.reshape(-1, 3).view(), aa.reshape(-1, 3).view()
        flat_v.flags.writeable = flat_a.flags.writeable = False
        result = _power_numeric(flat_v, flat_a, _iv_constants(p), _INTERVAL_POWERS)
        return float(result[0]) if vv.ndim == 1 else result.reshape(vv.shape[:-1])
    if ((v.ndim > 1 or a.ndim > 1)
            and (v.dtype != np.dtype('float64') or a.dtype != np.dtype('float64'))):
        vv, aa = np.broadcast_arrays(v, a)
        return np.array([shaft_power(x, y, p) for x, y in zip(vv.reshape(-1, 3), aa.reshape(-1, 3))]).reshape(vv.shape[:-1])
    theta = thrust(v, a, p)
    scalar = np.ndim(theta) == 0
    speed = small_norm(v) if scalar else _batch_norm(v)
    rho, area, c = p['rho'], p['rotor_area'], p['thrust_coeff']
    if scalar:
        profile = p['profile_delta']/8*(theta/(c*rho*area)+3*speed**2)*np.sqrt(rho*p['solidity']**2*area*theta/c)
        induced = (1+p['induced_epsilon'])*theta*np.sqrt(max(0., np.sqrt(theta**2/(4*rho**2*area**2)+speed**4/4)-speed**2/2))
        return float(profile+p['weight']*v[2]+rho*p['drag_area']/2*speed**3+induced)
    profile = p['profile_delta']/8*(theta/(c*rho*area)+3*np.float_power(speed, 2))*np.sqrt(rho*p['solidity']**2*area*theta/c)
    induced = (1+p['induced_epsilon'])*theta*np.sqrt(np.maximum(0., np.sqrt(np.float_power(theta, 2)/(4*rho**2*area**2)+np.float_power(speed, 4)/4)-np.float_power(speed, 2)/2))
    return profile+p['weight']*v[..., 2]+rho*p['drag_area']/2*np.float_power(speed, 3)+induced


@lru_cache(maxsize=128)
def _power_floor(items):
    p = dict(items)
    return p['power_floor_ratio']*shaft_power(np.zeros(3), np.zeros(3), p)


def power_floor(p):
    keys = ('weight', 'g0', 'rho', 'drag_area', 'rotor_area', 'thrust_coeff',
            'profile_delta', 'solidity', 'induced_epsilon', 'power_floor_ratio')
    return _power_floor(tuple((key, p[key]) for key in keys))


try:
    from numba import njit as _numeric_njit
except ImportError:
    _interval_numba = False
    def _numeric_njit(fn): return fn
else:
    _interval_numba = True

_INTERVAL_POWERS = np.array([2., 3., 4.])
_INTERVAL_POWERS.flags.writeable = False

@_numeric_njit
def _iv_rounded(lo,hi):
    eps=1e-12*max(abs(lo),abs(hi),2.2250738585072014e-308)
    return np.nextafter(lo-eps,-np.inf),np.nextafter(hi+eps,np.inf)
@_numeric_njit
def _iv_point(x):return x,x
@_numeric_njit
def _iv_add(x,y):return _iv_rounded(x[0]+y[0],x[1]+y[1])
@_numeric_njit
def _iv_mul(x,y):
    xs=(x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1])
    return _iv_rounded(min(xs),max(xs))
@_numeric_njit
def _iv_div(x,y):return _iv_mul(x,_iv_rounded(1/y[1],1/y[0]))
@_numeric_njit
def _iv_powiv(x,power):
    a,b=math.pow(x[0],power),math.pow(x[1],power)
    lower=0. if int(power)%2==0 and x[0]<=0<=x[1] else min(a,b)
    return _iv_rounded(lower,max(a,b))
@_numeric_njit
def _iv_sqrtiv(x):
    r=_iv_rounded(math.sqrt(max(0.,x[0])),math.sqrt(max(0.,x[1])))
    return max(0.,r[0]),r[1]
@_numeric_njit
def _iv_normiv(x,y,z,power):
    return _iv_sqrtiv(_iv_add(_iv_add(_iv_add(_iv_point(0.),_iv_powiv(x,power)),_iv_powiv(y,power)),_iv_powiv(z,power)))
@_numeric_njit
def _iv_shaft(v,a,t,c,powers):
    mass,weight,drag,profile_coef,thrust_den,profile_rad,c_thrust,induced_den,induced_coef=c
    vs=[_iv_add(_iv_point(v[i]),_iv_mul(t,_iv_point(a[i]))) for i in range(3)]
    speed=_iv_normiv(vs[0],vs[1],vs[2],powers[0])
    theta_parts=[_iv_add(_iv_mul(_iv_mul(speed,_iv_point(drag)),vs[i]),_iv_point(mass*a[i]+(weight if i==2 else 0.))) for i in range(3)]
    theta=_iv_normiv(theta_parts[0],theta_parts[1],theta_parts[2],powers[0])
    profile=_iv_mul(_iv_mul(_iv_add(_iv_div(theta,_iv_point(thrust_den)),_iv_mul(_iv_powiv(speed,powers[0]),_iv_point(3.))),_iv_point(profile_coef)),_iv_sqrtiv(_iv_div(_iv_mul(theta,_iv_point(profile_rad)),_iv_point(c_thrust))))
    root=_iv_sqrtiv(_iv_add(_iv_div(_iv_powiv(theta,powers[0]),_iv_point(induced_den)),_iv_div(_iv_powiv(speed,powers[2]),_iv_point(4.))))
    sub=_iv_div(_iv_powiv(speed,powers[0]),_iv_point(2.));rad=_iv_add(root,(-sub[1],-sub[0]))
    induced=_iv_mul(_iv_mul(theta,_iv_point(induced_coef)),_iv_sqrtiv((max(0.,rad[0]),max(0.,rad[1]))))
    return _iv_add(_iv_add(_iv_add(profile,_iv_mul(vs[2],_iv_point(weight))),_iv_mul(_iv_powiv(speed,powers[1]),_iv_point(drag))),induced)

def _iv_constants(p):
    rho,area,c=p['rho'],p['rotor_area'],p['thrust_coeff']
    return (p['weight']/p['g0'],p['weight'],rho*p['drag_area']/2,p['profile_delta']/8,c*rho*area,rho*p['solidity']**2*area,c,4*rho**2*area**2,1+p['induced_epsilon'])


@_numeric_njit
def _iv_separation(q, v, a, scale, t, powers):
    total = (0., 0.)
    for i in range(3):
        x = _iv_add(_iv_add(_iv_point(q[i]), _iv_mul(t, _iv_point(v[i]))),
                    _iv_div(_iv_mul(_iv_powiv(t, powers[0]), _iv_point(a[i])), _iv_point(2.)))
        total = _iv_add(total, _iv_powiv(_iv_div(x, _iv_point(scale[i])), powers[0]))
    return total


def _interval_kernel_inputs(vectors, t):
    return (_interval_numba and isinstance(t, Interval)
            and math.isfinite(t.lo) and math.isfinite(t.hi)
            and all(isinstance(x, np.ndarray) and x.shape == (3,) and x.dtype == np.float64
                    and np.all(np.isfinite(x)) for x in vectors))


@_numeric_njit
def _power_scalar(v, a, c, powers):
    for i in range(3):
        if not math.isfinite(v[i]) or not math.isfinite(a[i]):
            return False, 0.
    mass, weight, drag, profile_scale, theta_scale, profile_root, thrust_coeff, induced_denom, induced_scale = c
    speed = math.sqrt(np.dot(v, v))
    thrust = mass*a+np.array([0., 0., weight])+drag*speed*v
    theta = math.sqrt(np.dot(thrust, thrust))
    profile = profile_scale*(theta/theta_scale+3*math.pow(speed, powers[0]))*math.sqrt(profile_root*theta/thrust_coeff)
    induced = induced_scale*theta*math.sqrt(max(0., math.sqrt(math.pow(theta, powers[0])/induced_denom+math.pow(speed, powers[2])/4)-math.pow(speed, powers[0])/2))
    return True, profile+weight*v[2]+drag*math.pow(speed, powers[1])+induced


@_numeric_njit
def _power_numeric(v,a,c,powers):
    mass,weight,drag,profile_scale,theta_scale,profile_root,thrust_coeff,induced_denom,induced_scale=c
    result=np.empty(len(v))
    for i in range(len(v)):
        speed=math.sqrt(np.dot(v[i],v[i]))
        thrust=mass*a[i]+np.array([0.,0.,weight])+drag*speed*v[i]
        theta=math.sqrt(np.dot(thrust,thrust))
        profile=profile_scale*(theta/theta_scale+3*math.pow(speed,powers[0]))*math.sqrt(profile_root*theta/thrust_coeff)
        induced=induced_scale*theta*math.sqrt(max(0.,math.sqrt(math.pow(theta,powers[0])/induced_denom+math.pow(speed,powers[2])/4)-math.pow(speed,powers[0])/2))
        result[i]=profile+weight*v[i,2]+drag*math.pow(speed,powers[1])+induced
    return result

class _KernelInterval(Interval):
    @classmethod
    def rounded(cls, lo, hi):
        eps = 1e-12*max(abs(lo), abs(hi), 2.2250738585072014e-308)
        return cls(math.nextafter(lo-eps, -math.inf), math.nextafter(hi+eps, math.inf))

    def __neg__(self):
        return _KernelInterval(-self.hi, -self.lo)

    def sqrt(self):
        if self.hi < 0: raise ValueError('negative square root')
        r = self.rounded(math.sqrt(max(0, self.lo)), math.sqrt(max(0, self.hi)))
        return _KernelInterval(max(0, r.lo), r.hi)


def _kernel_iv(x):
    if isinstance(x, _KernelInterval): return x
    if isinstance(x, Interval): return _KernelInterval(x.lo, x.hi)
    return _KernelInterval(float(x), float(x))


def _kernel_norm(xs):
    return sum((_kernel_iv(x)**2 for x in xs), _kernel_iv(0)).sqrt()


def shaft_interval(v, a, p, t):
    if _interval_kernel_inputs((v, a), t):
        result = _iv_shaft(v, a, (float(t.lo), float(t.hi)), _iv_constants(p), _INTERVAL_POWERS)
        if math.isfinite(result[0]) and math.isfinite(result[1]): return Interval(*result)
    t = _kernel_iv(t) if isinstance(t, Interval) else t
    vs = [_kernel_iv(v[i])+t*a[i] for i in range(3)]
    speed = _kernel_norm(vs)
    theta = _kernel_norm([p['weight']/p['g0']*a[i]+(p['weight'] if i == 2 else 0)
                  +p['rho']*p['drag_area']/2*speed*vs[i] for i in range(3)])
    rho, area, c = p['rho'], p['rotor_area'], p['thrust_coeff']
    profile = p['profile_delta']/8*(theta/(c*rho*area)+3*speed**2)*(rho*p['solidity']**2*area*theta/c).sqrt()
    rad = (theta**2/(4*rho**2*area**2)+speed**4/4).sqrt()-speed**2/2
    induced = (1+p['induced_epsilon'])*theta*_KernelInterval(max(0, rad.lo), max(0, rad.hi)).sqrt()
    result = profile+p['weight']*vs[2]+rho*p['drag_area']/2*speed**3+induced
    return Interval(result.lo, result.hi)


def rate_interval(q, v, a, bs, buildings, power, sigma, p, t):
    if not sigma or power == 0:
        return iv(0)
    x = [iv(q[i])+t*v[i]+t**2*a[i]/2 for i in range(3)]
    d = norm([x[i]-bs[i] for i in range(3)])
    if d.lo <= 0:
        return Interval(0, float('inf'))
    gains = [p[f'beta_{tag}']*(d/p['d0'])**(-p[f'alpha_{tag}']) for tag in ('L', 'N')]
    definitely_los = all(any(max(x[i].hi, bs[i]) < b['lo'][i] or min(x[i].lo, bs[i]) > b['hi'][i]
                             for i in range(3)) for b in buildings)
    g = gains[0] if definitely_los else Interval(min(z.lo for z in gains), max(z.hi for z in gains))
    return sigma*p['bandwidth']*(1+power*g/(p['N0']*p['bandwidth'])).log2()


def array_signature(x):
    x = np.asarray(x)
    return x.dtype.str, x.shape, x.tobytes()


def shaft_parameters(p, extra=()):
    keys = ('weight', 'g0', 'rho', 'drag_area', 'rotor_area', 'thrust_coeff',
            'profile_delta', 'solidity', 'induced_epsilon')+extra
    return tuple((key, p[key]) for key in keys)


@lru_cache(maxsize=4096)
def _flight_energy_cached(vkey, akey, parameters, layer, settings, integrator, power):
    v = np.frombuffer(vkey[2], dtype=vkey[0]).reshape(vkey[1])
    a = np.frombuffer(akey[2], dtype=akey[0]).reshape(akey[1])
    p, cfg = dict(parameters), dict(settings)
    return p['delta']*p['P0']+integrator(lambda t: power(v+t*a, a, p),
                                       p['delta'], layer, cfg)/p['eta_flight']


def flight_energy(v, a, p, layer, cfg):
    settings = tuple((key, cfg.get(key)) for key in ('guard_nodes', 'quad_epsabs', 'quad_epsrel'))
    return _flight_energy_cached(array_signature(v), array_signature(a),
        shaft_parameters(p, ('delta', 'P0', 'eta_flight')), layer, settings, integrate, shaft_power)


def cpu_energy(F, p):
    return p['delta']*p['kappa']*F**3


def radio_energy(p_ul, sigma_ul, sigma_dl, p):
    return p['delta']*(sum(p_ul)/p['eta_tx']+p['P_tx']*sum(sigma_ul)+p['P_rx']*sum(sigma_dl))


def payload_energy(active, tasks, p):
    return p['delta']*sum(tasks[k]['P_'+phase] for k, phase in active)


def total_energy(parts):
    return sum(parts)


def battery(initial, energies):
    return initial-sum(energies)
