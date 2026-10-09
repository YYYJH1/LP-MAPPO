from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np
from lawn_mec.env import checker
from lawn_mec.env.generator import accept
from lawn_mec.env.geometry import Box, trajectory_box_residual
from lawn_mec.env.interval import enclose_integral, certify_lower
from lawn_mec.env.model import motion, flight_energy, shaft_interval, power_floor
from lawn_mec.env.params import checker_config
from .state import freeze
from .weakmemo import weak_method_lru_cache


class Uncertified(ValueError):
    pass


@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    q0: tuple
    v0: tuple
    q1: tuple
    v1: tuple
    accelerations: tuple
    edges: tuple
    energy: tuple
    energy_bounds: tuple
    certificate: tuple

    @property
    def duration(self):
        return len(self.accelerations)

    @property
    def E(self):
        return sum(self.energy)


def layers(instance):
    heights = tuple(85.+5.25*i for i in range(-3, 4))
    boxes = [Box.from_dict(t['S_'+phase]) for t in instance['tasks'] for phase in ('s', 'a')]
    boxes += [Box.from_dict(b).erode(instance['params']['clearance']) for b in instance['nodes']]
    if any(not all(b.lo[2] <= z <= b.hi[2] for z in heights) for b in boxes):
        raise Uncertified('seven layers do not fit the actual service band/node boxes')
    if 5.25 <= instance['params']['D_z']:
        raise Uncertified('layer pitch must strictly exceed D_z for audit separation')
    return heights


def trapezoid(q0, q1, speed, acceleration, delta, extra=0):
    displacement = np.asarray(q1)-q0
    distance = float(np.linalg.norm(displacement))
    if distance == 0:
        return []
    ramp = max(1, math.ceil(min(speed, math.sqrt(distance*acceleration))/(acceleration*delta)))
    cruise = max(0, math.ceil(distance/(speed*delta)-ramp))+extra
    peak = distance/((ramp+cruise)*delta)
    while peak/(ramp*delta) > acceleration or peak > speed:
        cruise += 1
        peak = distance/((ramp+cruise)*delta)
    a = displacement/distance*peak/(ramp*delta)
    return [a.copy() for _ in range(ramp)]+[np.zeros(3) for _ in range(cruise)]+[-a.copy() for _ in range(ramp)]


def hermite_accels(q0, v0, q1, v1, slots, dt):
    weights = np.array([np.full(slots, dt), (slots-np.arange(slots)-.5)*dt*dt])
    rhs = np.array([np.asarray(v1)-v0, np.asarray(q1)-q0-slots*dt*np.asarray(v0)])
    return (weights.T@np.linalg.solve(weights@weights.T, rhs)).tolist()


class PrimitiveLibrary:
    speed_grid = (4., 6., 8., 10., 12., 14., 16., 18., 20.)
    vertical_grid = (1., 2., 3., 4., 5.)
    loiter_radii = (6., 9., 12.)
    loiter_periods = (12, 16, 20, 24, 32)

    def __init__(self, instance, cfg=None):
        tic = perf_counter()
        inst = dict(instance)
        inst['_geometry_accepted'] = accept(instance)['passed']
        if not inst['_geometry_accepted']:
            raise Uncertified('instance geometry acceptance failed')
        self.instance, self.p = freeze(inst), freeze(inst['params'])
        self.cfg = freeze(cfg or checker_config())
        self.heights = layers(instance)
        self.centres = [Box.from_dict(b).center for b in instance['nodes']]
        self.self_edges = {e['source']: i for i, e in enumerate(instance['edges']) if e['source'] == e['target']}
        self.edge_ids = {(e['source'], e['target']): i for i, e in enumerate(instance['edges'])}
        self.cache = {}
        self.rejected = {}
        self.build_time_s = perf_counter()-tic

    def point(self, node, height):
        q = self.centres[node].copy(); q[2] = height
        return q

    @weak_method_lru_cache(maxsize=32768)
    def dynamics(self, velocity, acceleration):
        v, a = np.array(velocity), np.array(acceleration)
        speed = checker.flight_speed(v, a, self.p)
        accel = max(np.linalg.norm(a[:2])-self.p['a_h'], abs(a[2])-self.p['a_z'])
        power = checker.power_check(v, a, self.p, self.cfg, 'audit')
        margin = certify_lower(lambda t: shaft_interval(v, a, self.p, t), self.p['delta'],
                               power_floor(self.p)+2., self.cfg['interval_budget'])
        if speed > self.cfg['epsilon_num'] or accel > self.cfg['epsilon_num'] or power['status'] != 'pass' or margin['status'] != 'pass':
            raise Uncertified(dict(speed=speed, acceleration=accel, power=power, margin_2W=margin))
        energy = flight_energy(v, a, self.p, 'evaluation', self.cfg)
        bound = self.p['delta']*self.p['P0']+enclose_integral(
            lambda t: shaft_interval(v, a, self.p, t), self.p['delta'], self.cfg['integral_subdivisions'])/self.p['eta_flight']
        return energy, tuple(bound.as_list()), freeze(dict(speed=speed, acceleration=accel, power=power, margin_2W=margin))

    def certify(self, name, family, q0, v0, accelerations, edges, *, service=None, limit=None):
        q, v = np.array(q0, float), np.array(v0, float)
        energies, bounds, certs = [], [], []
        if len(accelerations) != len(edges):
            raise ValueError('one explicit edge per primitive slot required')
        for a, edge in zip(accelerations, edges):
            a = np.asarray(a, float)
            c04, inner = checker.corridor_check(q, v, a, self.instance['edges'][edge], self.instance, self.cfg, 'audit')
            if c04['status'] != 'pass':
                raise Uncertified(dict(C04=c04, inner=inner))
            energy, bound, dynamic = self.dynamics(tuple(v), tuple(a))
            certificate = dict(C04=c04, dynamic=dynamic)
            if service is not None:
                residual = trajectory_box_residual(q, v, a, service, self.p['delta'])
                speed = checker.service_speed(v, a, limit, self.p['delta'])
                if max(residual, speed) > self.cfg['epsilon_num']:
                    raise Uncertified(dict(C05=residual, C06=speed))
                certificate.update(C05=residual, C06=speed)
            energies.append(energy); bounds.append(bound); certs.append(freeze(certificate))
            q, v = motion(q, v, a, self.p['delta'])
        return Primitive(name, family, tuple(q0), tuple(v0), tuple(q), tuple(v),
                         tuple(tuple(float(x) for x in a) for a in accelerations), tuple(edges),
                         tuple(energies), tuple(bounds), tuple(certs))

    def line_edges(self, q0, accelerations, source, target):
        q, v = np.array(q0, float), np.zeros(3)
        arrived = source == target
        edges = []
        box = Box.from_dict(self.instance['nodes'][target])
        for i, a in enumerate(accelerations):
            if i and box.contains(q, self.cfg['epsilon_num']):
                arrived = True
            edges.append(self.self_edges[target] if arrived else self.edge_ids[source, target])
            q, v = motion(q, v, a, self.p['delta'])
        return edges

    def straight(self, source, target, height):
        key = ('straight', source, target, height)
        if key in self.cache:
            return self.cache[key]
        q0, q1 = self.point(source, height), self.point(target, height)
        options = []
        for speed in self.speed_grid:
            if speed > self.p['V_h']:
                continue
            for extra in range(5):
                aa = trapezoid(q0, q1, speed, self.p['a_h'], self.p['delta'], extra)
                try:
                    prim = self.certify(f'line:{source}:{target}:{height}:{speed}:{extra}', 'cruise', q0, (0.,)*3,
                                        aa, self.line_edges(q0, aa, source, target))
                except Uncertified as e:
                    self.rejected[('line', source, target, height, speed, extra)] = str(e)
                    continue
                options.append(prim)
                break
        if not options:
            raise Uncertified(f'no straight primitive for {key}')
        self.cache[key] = tuple(options)
        return self.cache[key]

    def vertical(self, node, start, end):
        key = ('vertical', node, start, end)
        if key in self.cache:
            return self.cache[key]
        options = []
        for speed in self.vertical_grid:
            if speed > self.p['V_up' if end >= start else 'V_down']:
                continue
            aa = trapezoid(self.point(node, start), self.point(node, end), speed, self.p['a_z'], self.p['delta'])
            try:
                options.append(self.certify(f'vertical:{node}:{start}:{end}:{speed}', 'vertical', self.point(node, start),
                                            (0.,)*3, aa, [self.self_edges[node]]*len(aa)))
            except Uncertified as e:
                self.rejected[key+(speed,)] = str(e)
        if not options:
            raise Uncertified(f'no vertical primitive for {key}')
        self.cache[key] = tuple(options)
        return self.cache[key]

    def connector(self, node, height, q0, v0, q1, v1, family='loiter'):
        for count in range(2, 25):
            aa = hermite_accels(q0, v0, q1, v1, count, self.p['delta'])
            try:
                return self.certify(f'connector:{node}:{height}:{count}:{tuple(v0)}:{tuple(v1)}', family,
                                    q0, v0, aa, [self.self_edges[node]]*count)
            except Uncertified:
                continue
        raise Uncertified('no connector in the enumerated 2..24-slot family')

    def corner(self, node, height, incoming, outgoing, speed=2.):
        key = ('corner', node, height, tuple(incoming), tuple(outgoing), speed)
        if key not in self.cache:
            q = self.point(node, height)
            vin, vout = speed*np.asarray(incoming), speed*np.asarray(outgoing)
            self.cache[key] = self.connector(node, height, q-vin, vin, q+vout, vout, 'cruise')
        return self.cache[key]

    def hover(self, node, height):
        key = ('hover', node, height)
        if key not in self.cache:
            self.cache[key] = self.certify(f'hover:{node}:{height}', 'hover', self.point(node, height), (0.,)*3,
                                           [(0.,)*3], [self.self_edges[node]])
        return self.cache[key]

    def loiters(self, node, height):
        key = ('loiters', node, height)
        if key in self.cache:
            return self.cache[key]
        q = self.point(node, height); options = []
        for radius in self.loiter_radii:
            for period in self.loiter_periods:
                angle = 2*math.pi/period
                speed = 2*radius*math.tan(angle/2)/self.p['delta']
                point = q+np.array([radius, 0., 0.])
                velocity = np.array([0., speed, 0.])
                vv = [speed*np.array([-math.sin(i*angle), math.cos(i*angle), 0.]) for i in range(period)]
                aa = [(vv[(i+1) % period]-vv[i])/self.p['delta'] for i in range(period)]
                try:
                    cycle = self.certify(f'circle:{node}:{height}:{radius}:{period}', 'loiter', point, velocity,
                                         aa, [self.self_edges[node]]*period)
                    entry = self.connector(node, height, q, np.zeros(3), point, velocity)
                    leave = self.connector(node, height, point, velocity, q, np.zeros(3))
                except Uncertified as e:
                    self.rejected[key+(radius, period)] = str(e)
                    continue
                options.append((entry, cycle, leave))
        self.cache[key] = tuple(options)
        return self.cache[key]

    def service(self, k, phase, height):
        key = ('service', k, phase, height)
        if key in self.cache:
            return self.cache[key]
        task = self.instance['tasks'][k]; node = task['z_'+phase]
        box = Box.from_dict(task['S_'+phase]); count, limit = task['L_'+phase], task['V_'+phase]
        q = self.point(node, height); options = []
        for fraction in (0., .25, .5, 1.):
            aa = [np.zeros(3) for _ in range(count)]
            if fraction:
                if count < 3:
                    continue
                a = min(limit/self.p['delta'], self.p['a_h']/2)*fraction
                aa[0][0], aa[1][0], aa[2][0] = a, -2*a, a
            try:
                options.append(self.certify(f'service:{k}:{phase}:{height}:{fraction}', 'service', q, (0.,)*3,
                                            aa, [self.self_edges[node]]*count, service=box, limit=limit))
            except Uncertified as e:
                self.rejected[key+(fraction,)] = str(e)
        if not options:
            raise Uncertified(f'no service primitive for {key}')
        self.cache[key] = tuple(options)
        return self.cache[key]

    def build(self, *, all_nodes=True):
        tic = perf_counter()
        nodes = range(len(self.centres)) if all_nodes else (0,)
        heights = sorted(set(self.heights+(85.,)))
        for node in nodes:
            for h in self.heights:
                self.hover(node, h); self.loiters(node, h)
                for turn in ((0., 1., 0.), (0., -1., 0.), (-1., 0., 0.)):
                    for speed in (0., 2., 4.):
                        self.corner(node, h, (1., 0., 0.), turn, speed)
            for start in heights:
                for end in heights:
                    if start != end:
                        self.vertical(node, start, end)
        for source, target in self.edge_ids:
            if source != target:
                for h in self.heights:
                    self.straight(source, target, h)
        for k in range(len(self.instance['tasks'])):
            for phase in ('s', 'a'):
                for h in self.heights:
                    self.service(k, phase, h)
        self.build_time_s += perf_counter()-tic
        return self
