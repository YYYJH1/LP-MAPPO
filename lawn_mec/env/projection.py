import numpy as np
from .geometry import trajectory_box_residual, face_accel_bound


def disk(x, radius):
    n = np.linalg.norm(x)
    return x*min(1., radius/n) if n else x.copy()


def degraded(v, p):
    v = np.asarray(v)
    return np.r_[-disk(v[:2]/p['delta'], p['a_h']), np.clip(-v[2]/p['delta'], -p['a_z'], p['a_z'])]


def project(candidate, q, v, box, p, cfg, service_box=None, service_speed=None):
    q, v, x = np.asarray(q), np.asarray(v), np.asarray(candidate, float).copy()
    dt = p['delta']; boxes = [box]+([] if service_box is None else [service_box])
    def accel(z): return np.r_[disk(z[:2], p['a_h']), np.clip(z[2], -p['a_z'], p['a_z'])]
    def speed(z):
        w = v+dt*z
        return (np.r_[disk(w[:2], p['V_h']), np.clip(w[2], -p['V_down'], p['V_up'])]-v)/dt
    projections = [accel, speed]
    if service_speed is not None:
        projections.append(lambda z: (disk(v+dt*z, service_speed)-v)/dt)
    impossible = False
    for region in boxes:
        for h, b in region.faces():
            bound = face_accel_bound(h@q-b, h@v, dt)
            if not np.isfinite(bound): impossible = True; continue
            projections.append(lambda z, h=h, bound=bound: z-max(0., h@z-bound)*h)
    corrections = [np.zeros(3) for _ in projections]
    for iteration in range(cfg['projection_iterations']):
        old = x.copy()
        for i, fn in enumerate(projections):
            y = x+corrections[i]; x = fn(y); corrections[i] = y-x
        if np.linalg.norm(x-old) <= cfg['projection_tolerance']:
            if all(np.linalg.norm(fn(x)-x) <= cfg['projection_tolerance'] for fn in projections): break
    residuals = [np.linalg.norm(fn(x)-x) for fn in projections]
    residuals += [max(0., trajectory_box_residual(q, v, x, region, dt)) for region in boxes]
    residuals += [max(0., np.linalg.norm(v[:2])-p['V_h']), max(0., v[2]-p['V_up']), max(0., -v[2]-p['V_down'])]
    if service_speed is not None: residuals.append(max(0., np.linalg.norm(v)-service_speed))
    ok = not impossible and max(residuals, default=0) <= cfg['epsilon_num']
    return {'accel': x if ok else degraded(v, p), 'success': ok, 'iterations': iteration+1,
            'residual': float(max(residuals, default=0)), 'candidate_in_set': not impossible and all(
                np.linalg.norm(fn(np.asarray(candidate))-candidate) <= cfg['epsilon_num'] for fn in projections)}
