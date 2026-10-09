from dataclasses import dataclass
from heapq import heappush, heappop
from time import perf_counter
import numpy as np
from .primitives import PrimitiveLibrary, Uncertified


@dataclass(frozen=True)
class Target:
    kind: str
    node: int | None = None
    task: int | None = None
    phase: str | None = None
    uav: int | None = None


@dataclass(frozen=True)
class Plan:
    status: str
    arrival_slot: int
    earliest_slot: int
    chain: tuple = ()
    path: tuple = ()
    energy: float = 0.
    call_time_s: float = 0.

    @property
    def duration(self):
        return sum(p.duration for p in self.chain)

    @property
    def certificate(self):
        return tuple(c for p in self.chain for c in p.certificate)

    def actions(self):
        return [dict(edge=e, accel=list(a), family=p.family, primitive=p.name)
                for p in self.chain for e, a in zip(p.edges, p.accelerations)]


class MotionPlanner:
    def __init__(self, library: PrimitiveLibrary, max_alternatives=4):
        if not isinstance(max_alternatives, int) or not 0 <= max_alternatives <= 4:
            raise ValueError('max_alternatives must be in 0..4')
        self.lib = library
        self.max_alternatives = max_alternatives
        self.path_cache, self.wait_cache, self.travel_cache = {}, {}, {}
        self.table_build_s = 0.
        self.graph = {node: [] for node in range(len(library.centres))}
        for source, target in library.edge_ids:
            if source != target:
                self.graph[source].append(target)
        for neighbours in self.graph.values():
            neighbours.sort()
        self.N = round(library.p['T']/library.p['delta'])
        self.landing_duration = max(p.duration for h in library.heights
                                    for p in library.vertical(0, h, 85.))
        self.landing_start = self.N-self.landing_duration

    def paths(self, source, target):
        key = source, target
        if key not in self.path_cache:
            queue, paths = [(0., (source,))], []
            while queue and len(paths) < self.max_alternatives+1:
                distance, path = heappop(queue)
                if path[-1] == target:
                    paths.append(path)
                    continue
                for node in self.graph[path[-1]]:
                    if node not in path:
                        length = float(np.linalg.norm(self.lib.centres[node]-self.lib.centres[path[-1]]))
                        heappush(queue, (distance+length, path+(node,)))
            self.path_cache[key] = tuple(paths)
        return self.path_cache[key]

    def wait_table(self, node, height, horizon):
        key = node, height
        if key in self.wait_cache and len(self.wait_cache[key][0]) > horizon:
            return self.wait_cache[key]
        tic = perf_counter()
        hover = self.lib.hover(node, height)
        options = [(hover,)]
        for entry, cycle, leave in self.lib.loiters(node, height):
            for repeats in range(1, (horizon-entry.duration-leave.duration)//cycle.duration+1):
                options.append((entry,)+(cycle,)*repeats+(leave,))
        durations = np.array([sum(p.duration for p in chain) for chain in options], int)
        energies = np.array([sum(p.E for p in chain) for chain in options])
        costs = np.full(horizon+1, np.inf); costs[0] = 0.
        choices = np.full(horizon+1, -1, int)
        for t in range(1, horizon+1):
            possible = np.flatnonzero(durations <= t)
            values = costs[t-durations[possible]]+energies[possible]
            j = int(np.argmin(values)); choices[t] = possible[j]; costs[t] = values[j]
        result = costs, choices, tuple(options), durations
        self.wait_cache[key] = result
        self.table_build_s += perf_counter()-tic
        return result

    def waiting(self, node, height, duration):
        if duration < 0:
            raise ValueError('negative waiting duration')
        costs, choices, options, durations = self.wait_table(node, height, duration)
        blocks, remaining = [], duration
        while remaining:
            j = choices[remaining]; blocks.append(options[j]); remaining -= int(durations[j])
        chain = tuple(p for block in reversed(blocks) for p in block)
        return costs[duration], chain

    def travel_table(self, path, height, horizon):
        key = path, height
        if key in self.travel_cache and self.travel_cache[key][0] >= horizon:
            return self.travel_cache[key][1]
        tic = perf_counter()
        states = {0: (0., ())}
        for a, b in zip(path[:-1], path[1:]):
            following = {}
            for elapsed, (cost, chain) in sorted(states.items()):
                for primitive in self.lib.straight(a, b, height):
                    t, e = elapsed+primitive.duration, cost+primitive.E
                    if t <= horizon and (t not in following or e < following[t][0]):
                        following[t] = e, chain+(primitive,)
            states = following
        self.travel_cache[key] = horizon, states
        self.table_build_s += perf_counter()-tic
        return states

    def _plan_rest(self, source, target, height, duration, route_path=None):
        best = None
        for path in (self.paths(source, target) if route_path is None else (tuple(route_path),)):
            travel = self.travel_table(path, height, duration)
            waits = [self.wait_table(node, height, duration) for node in path]
            for elapsed, (travel_energy, legs) in sorted(travel.items()):
                if elapsed > duration:
                    continue
                slack = duration-elapsed
                wait_node_index = min(range(len(path)), key=lambda j: (waits[j][0][slack], j))
                e = travel_energy+waits[wait_node_index][0][slack]
                if best is None or e < best[0]:
                    _, wait_chain = self.waiting(path[wait_node_index], height, slack)
                    chain = legs[:wait_node_index]+wait_chain+legs[wait_node_index:]
                    best = e, chain, path
        return best

    def earliest_duration(self, source, target, height):
        return min(sum(min(p.duration for p in self.lib.straight(a, b, height))
                       for a, b in zip(path[:-1], path[1:])) for path in self.paths(source, target))

    def plan(self, q, v, start_slot, target: Target, T, height, *, route_path=None):
        tic = perf_counter()
        if height not in self.lib.heights or not isinstance(start_slot, (int, np.integer)) or not isinstance(T, (int, np.integer)):
            raise ValueError('canonical layer and integer slot indices required')
        if not 0 <= start_slot <= self.N or not 0 <= T <= self.N:
            raise ValueError('slot outside the instance horizon')
        source = next((node for node in self.graph
                       if np.linalg.norm(np.asarray(q)-self.lib.point(node, height)) <= self.lib.cfg['epsilon_num']), None)
        if source is None:
            raise ValueError('start must be at a node centre on the current layer')
        prefix = ()
        if np.linalg.norm(v) > self.lib.cfg['epsilon_num']:
            admitted = [speed*np.eye(3)[axis]*sign for speed in (2., 4.) for axis in (0, 1) for sign in (-1, 1)]
            if not any(np.linalg.norm(np.asarray(v)-w) <= self.lib.cfg['epsilon_num'] for w in admitted):
                raise ValueError('velocity is outside the certified cruise-exit grid')
            prefix = (self.lib.connector(source, height, np.asarray(q), np.asarray(v), self.lib.point(source, height), np.zeros(3), 'cruise'),)
        prefix_time = sum(p.duration for p in prefix)
        if target.kind == 'service':
            if target.phase not in ('s', 'a') or target.task is None or not 0 <= target.task < len(self.lib.instance['tasks']):
                raise ValueError('service target needs a valid task and phase')
            task = self.lib.instance['tasks'][target.task]
            dest = task['z_'+target.phase]
            if T+task['L_'+target.phase] > self.N:
                raise ValueError('service target cannot finish within the horizon')
            self.lib.service(target.task, target.phase, height)
        elif target.kind == 'node':
            dest = target.node
        elif target.kind == 'endpoint':
            if target.uav is None or not 0 <= target.uav < self.lib.p['U']:
                raise ValueError('endpoint target needs UAV index')
            final = np.asarray(self.lib.instance['qF'][target.uav])
            dest = next((node for node in self.graph if np.linalg.norm(final-self.lib.point(node, 85.)) <= self.lib.cfg['epsilon_num']), None)
            if dest is None:
                raise ValueError('endpoint must be a node centre at 85 m')
            if not self.landing_start <= T <= self.N:
                raise ValueError('endpoint target is outside the common landing window')
        else:
            raise ValueError('unknown target kind')
        if dest not in self.graph:
            raise ValueError('invalid target node')
        if route_path is not None and tuple(route_path) not in self.paths(source, dest):
            raise ValueError('route_path must be an exposed certified candidate path')
        travel_min = (self.earliest_duration(source, dest, height) if route_path is None else
                      sum(min(p.duration for p in self.lib.straight(a, b, height))
                          for a, b in zip(route_path[:-1], route_path[1:])))
        earliest = start_slot+prefix_time+travel_min
        deadline = T
        terminal = ()
        if target.kind == 'endpoint':
            deadline = self.landing_start
            verticals = self.lib.vertical(dest, height, 85.)
            allowed = [p for p in verticals if p.duration <= T-deadline]
            earliest = max(earliest, deadline)+min(p.duration for p in verticals)
            if start_slot+prefix_time+travel_min > deadline:
                return Plan('landing_window_unreachable', earliest, earliest, call_time_s=perf_counter()-tic)
            if not allowed:
                return Plan('too_early', earliest, earliest, call_time_s=perf_counter()-tic)
            hover = self.lib.hover(dest, height)
            vertical = min(allowed, key=lambda p: (p.E+(T-deadline-p.duration)*hover.E, p.name))
            terminal = (hover,)*(T-deadline-vertical.duration)+(vertical,)
        if T < earliest or deadline < start_slot+prefix_time:
            return Plan('too_early', earliest, earliest, call_time_s=perf_counter()-tic)
        result = self._plan_rest(source, dest, height, deadline-start_slot-prefix_time, route_path)
        if result is None:
            return Plan('too_early', earliest, earliest, call_time_s=perf_counter()-tic)
        energy, middle, path = result
        chain = prefix+middle+terminal
        return Plan('ok', T, earliest, chain, path, sum(p.E for p in chain), perf_counter()-tic)

    def precompute(self):
        tic = perf_counter(); size = len(self.graph)
        times = np.zeros((len(self.lib.heights), size, size), dtype=np.int32)
        energies = np.zeros_like(times, dtype=float)
        for layer, height in enumerate(self.lib.heights):
            for source in self.graph:
                for target in self.graph:
                    times[layer, source, target] = self.earliest_duration(source, target, height)
                    energies[layer, source, target] = min(
                        sum(min(p.E for p in self.lib.straight(a, b, height)) for a, b in zip(path[:-1], path[1:]))
                        for path in self.paths(source, target))
        self.minimum_times, self.minimum_energies = times, energies
        return dict(build_time_s=perf_counter()-tic, table_bytes=times.nbytes+energies.nbytes,
                    shape=list(times.shape))
