from time import perf_counter
import math
import numpy as np
from lawn_mec.env.geometry import Box
from lawn_mec.env.model import rate
from lawn_mec.exec_v2.events import EventExecutor
from .reference_rate import DEFAULT_REFERENCE_RATE, check_reference_rate


def bins(lo, hi):
    width = max(0, int(hi)-int(lo)+1)
    return tuple((int(lo)+b*width//8, int(lo)+(b+1)*width//8-1) for b in range(8))


def bin_of(windows, slot):
    for b, (lo, hi) in enumerate(windows):
        if lo <= slot <= hi:
            return b
    raise ValueError(f'slot {slot} is outside bins {windows}')


def finish(work, capacities, start, missing):
    left = float(work)
    for n in range(int(start), len(capacities)):
        left -= capacities[n]
        if left <= 0:
            return n+1
    return missing


class CommitmentTables:
    reference_rate = DEFAULT_REFERENCE_RATE

    def __init__(self, instance, *, executor=None, reference_rate=DEFAULT_REFERENCE_RATE):
        check_reference_rate(reference_rate)
        tic = perf_counter()
        self.instance = instance
        self.ex = executor or EventExecutor(instance)
        self.lib, self.planner = self.ex.lib, self.ex.planner
        self.p, self.tasks = instance['params'], instance['tasks']
        self.N, self.D = self.ex.state.N, self.planner.landing_start
        self.min_time = np.min(self.planner.minimum_times, axis=0)
        self.own = [tuple(dict.fromkeys(k for k, _ in seq)) for seq in instance['visits']]
        self.windows, self.visit_bins = {}, {}
        for u, seq in enumerate(instance['visits']):
            earliest, node = self.ex.takeoff_end, self.ex.initial_nodes[u]
            suffix, dest = 0, self.ex.final_nodes[u]
            latest = {}
            for k, phase in reversed(seq):
                t = self.tasks[k]; source = t['z_'+phase]
                suffix += int(self.min_time[source, dest])+t['L_'+phase]
                latest[k, phase] = self.D-suffix
                dest = source
            for k, phase in seq:
                t = self.tasks[k]; dest = t['z_'+phase]
                earliest += int(self.min_time[node, dest])
                self.windows[k, phase] = (earliest, latest[k, phase])
                self.visit_bins[k, phase] = bins(earliest, latest[k, phase])
                earliest += t['L_'+phase]; node = dest
        self.radio_capacity = self._optimistic_radio()
        self.ready_by_location = np.full((len(self.tasks), self.p['M']+1, self.N+1), self.N+1, dtype=np.int32)
        self.upload_ready = np.full((len(self.tasks), self.p['M'], self.N+1), self.N+1, dtype=np.int32)
        self.cpu_capacity = np.array(self.p['delta']*(self.p['G_total']-np.asarray(instance['load'], float)))
        if np.any(self.cpu_capacity < 0):
            raise ValueError('negative MEC capacity')
        for k, t in enumerate(self.tasks):
            for m in range(1, self.p['M']+1):
                for start in range(self.N+1):
                    ul = start+math.ceil(t['D']/self.radio_capacity[m-1, 0])
                    cpu = finish(t['C'], self.cpu_capacity[m-1], ul, self.N+1)
                    self.upload_ready[k, m-1, start] = min(self.N+1, cpu+math.ceil(t['O']/self.radio_capacity[m-1, 1]))
            for ts in range(self.N+1):
                release = ts+t['L_s']
                self.ready_by_location[k, 0, ts] = min(self.N+1, release+math.ceil(t['C']/(self.p['delta']*self.p['F_local'])))
                if release <= self.N:
                    self.ready_by_location[k, 1:, ts] = self.upload_ready[k, :, release]
        self.ready_min = np.min(self.ready_by_location, axis=1)
        if np.any(np.diff(self.ready_min, axis=1) < 0):
            raise AssertionError('C2 readiness must be nondecreasing')
        self.upload_windows, self.upload_bins = {}, {}
        for k, t in enumerate(self.tasks):
            lo = self.windows[k, 's'][0]+t['L_s']
            possible = np.flatnonzero(np.min(self.upload_ready[k], axis=0) <= self.windows[k, 'a'][1])
            hi = int(possible[-1]) if len(possible) else lo-1
            self.upload_windows[k] = (lo, hi)
            self.upload_bins[k] = bins(lo, hi)
        self._energy = {}
        if reference_rate != DEFAULT_REFERENCE_RATE:
            self.reference_rate = reference_rate
            self.certified_upload = self._certified_uploads()
        self.build_s = perf_counter()-tic

    def _certified_uploads(self):
        from lawn_mec.exec_v2.state import SimState
        physics = SimState(self.instance, layer='audit').physics
        zero = np.zeros(3); result = {}
        for k, t in enumerate(self.tasks):
            for m in range(1, self.p['M']+1):
                for layer, h in enumerate(self.lib.heights):
                    q = np.asarray(self.lib.point(t['z_s'], h), float)
                    cap = max(0., physics.radio(q, zero, zero, m-1, self.p['P_ul'], 1.).lo)
                    result[k, m, layer] = self.N+1 if cap <= 0 else max(1, math.ceil(t['D']/cap))
        return result

    def _optimistic_radio(self):
        boxes = [Box.from_dict(b).erode(self.p['clearance']) for b in self.instance['nodes']]
        boxes += [Box.from_dict(e['box']).erode(self.p['clearance']) for e in self.instance['edges'] if 'box' in e]
        for e in self.instance['edges']:
            if 'box' not in e:
                a, b = (Box.from_dict(self.instance['nodes'][e[key]]) for key in ('source', 'target'))
                boxes.append(Box(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi)).erode(self.p['clearance']))
        result = []
        for bs in self.instance['bs']:
            d = min(b.point_distance(bs) for b in boxes)
            dmax = max(np.linalg.norm(np.maximum(np.abs(b.lo-bs), np.abs(b.hi-bs))) for b in boxes)
            if d <= 0:
                raise ValueError('BS intersects flight-domain bounding boxes')
            for distance in (d, dmax):
                los = self.p['beta_L']*(distance/self.p['d0'])**(-self.p['alpha_L'])
                nlos = self.p['beta_N']*(distance/self.p['d0'])**(-self.p['alpha_N'])
                if nlos > los:
                    raise ValueError('C2 LoS bound does not dominate NLoS')
            gain = self.p['beta_L']*(d/self.p['d0'])**(-self.p['alpha_L'])
            result.append([math.nextafter(self.p['delta']*self.p['bandwidth']*math.log2(1+power*gain/(self.p['N0']*self.p['bandwidth'])), math.inf) for power in (self.p['P_ul'], self.p['P_dl'])])
        return np.array(result)

    def travel_energy(self, source, dest, layer=None):
        key = source, dest, layer
        if key in self._energy:
            return self._energy[key]
        cost = np.full(self.N+1, np.inf)
        heights = self.lib.heights if layer is None else (self.lib.heights[layer],)
        for h in heights:
            for path in self.planner.paths(source, dest):
                waiting = np.minimum.reduce([self.planner.wait_table(node, h, self.N)[0][:self.N+1] for node in path])
                for duration, (energy, _) in self.planner.travel_table(path, h, self.N).items():
                    cost[duration:] = np.minimum(cost[duration:], energy+waiting[:self.N+1-duration])
        self._energy[key] = cost
        return cost

    def takeoff_energy(self, u):
        return min(p.E+(self.ex.takeoff_end-p.duration)*self.lib.hover(self.ex.initial_nodes[u], self.lib.heights[ell]).E for (owner, ell), p in self.ex.takeoffs.items() if owner == u)

    def landing_energy(self, u):
        return min(p.E+(self.N-self.D-p.duration)*self.lib.hover(self.ex.final_nodes[u], h).E for h in self.lib.heights for p in self.lib.vertical(self.ex.final_nodes[u], h, 85.) if p.duration <= self.N-self.D)

    def node_upload(self, k, m, layer):
        if self.reference_rate != DEFAULT_REFERENCE_RATE:
            return self.certified_upload[k, m, layer]
        t = self.tasks[k]; q = self.lib.point(t['z_s'], self.lib.heights[layer])
        cap = self.p['delta']*rate(q, self.instance['bs'][m-1], self.instance['buildings'], self.p['P_ul'], 1., self.p)
        return max(1, math.ceil(t['D']/cap))
