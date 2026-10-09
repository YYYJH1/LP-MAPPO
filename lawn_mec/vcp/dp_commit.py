from dataclasses import dataclass
from itertools import product
from heapq import heappush, heappop
from time import perf_counter
import math
import numpy as np
from numba import njit
from .tables import bin_of, finish


@njit(cache=True)
def advance(previous, kernel, length, lo, hi):
    N = len(previous)-1
    out = np.full(N+1, np.inf)
    parents = np.full(N+1, -1, np.int32)
    for start in range(max(0, lo), min(hi, N-length)+1):
        for before in range(start+1):
            value = previous[before]+kernel[start-before]
            if value < out[start+length]:
                out[start+length] = value
                parents[start+length] = before
    return out, parents


@dataclass(frozen=True)
class Schedule:
    uav: int
    keep: tuple
    energy: float
    starts: tuple
    truncation: int


class RelaxedDP:
    def __init__(self, tables):
        tic = perf_counter()
        self.tables = tables
        self.schedules = []
        for u, own in enumerate(tables.own):
            self.schedules.append({mask: self._solve(u, tuple(k for j, k in enumerate(own) if mask & (1 << j))) for mask in range(1 << len(own))})
        self.build_s = perf_counter()-tic
        self.reference_cache = {}
        self.reference_diagnostics = {}

    def _solve(self, u, keep):
        tb = self.tables; seq = tb.instance['visits'][u]
        previous = np.full(tb.N+1, np.inf)
        previous[tb.ex.takeoff_end] = tb.takeoff_energy(u)
        node = tb.ex.initial_nodes[u]; end = tb.ex.final_nodes[u]
        parents, lengths = [], []
        last_required = max((j+1 for j, (k, phase) in enumerate(seq) if k in keep and phase == 'a'), default=0)
        best = (math.inf, 0, 0)
        for pointer in range(len(seq)+1):
            if pointer >= last_required:
                kernel = tb.travel_energy(node, end)
                values = previous[:tb.D+1]+kernel[tb.D::-1]+tb.landing_energy(u)
                time = int(np.argmin(values))
                if values[time] < best[0]:
                    best = float(values[time]), pointer, time
            if pointer == len(seq):
                break
            k, phase = seq[pointer]; task = tb.tasks[k]; dest = task['z_'+phase]
            lo, hi = 0, tb.D-task['L_'+phase]
            if k in keep and phase == 'a':
                lo = int(tb.ready_min[k, min(tb.N, tb.windows[k, 's'][0])])
                hi = min(hi, math.floor(task['d']/tb.p['delta'])-task['L_a'])
            costs, parent = advance(previous, tb.travel_energy(node, dest), task['L_'+phase], lo, hi)
            service = min(p.E for h in tb.lib.heights for p in tb.lib.service(k, phase, h))
            previous = costs+service
            parents.append(parent); lengths.append(task['L_'+phase]); node = dest
        energy, pointer, time = best
        starts = []
        if math.isfinite(energy):
            for j in range(pointer-1, -1, -1):
                starts.append(time-lengths[j]); time = int(parents[j][time])
            starts.reverse()
        return Schedule(u, keep, energy, tuple(starts), pointer)

    def precheck(self, keep_sets):
        if len(keep_sets) != len(self.schedules):
            raise ValueError('one keep set per UAV required')
        reasons = []
        for u, keep in enumerate(keep_sets):
            if not set(keep) <= set(self.tables.own[u]):
                raise ValueError('foreign task in keep set')
            mask = sum(1 << j for j, k in enumerate(self.tables.own[u]) if k in keep)
            energy = self.schedules[u][mask].energy
            battery = self.tables.p['battery']
            allowance = 1e-9*max(1., abs(battery))
            reasons.append(dict(uav=u, energy_J=energy, battery_J=battery, rejected=energy > battery+allowance))
        return dict(passed=not any(r['rejected'] for r in reasons), uavs=reasons)

    def ranked(self, u):
        return sorted((s for s in self.schedules[u].values() if s.energy <= self.tables.p['battery']), key=lambda s: (-len(s.keep), s.energy, s.keep))

    def best(self):
        rows = [self.ranked(u) for u in range(len(self.schedules))]
        if any(not row for row in rows):
            raise ValueError('no battery-feasible relaxed keep set for a UAV')
        return tuple(row[0] for row in rows)

    def combinations(self, K=16):
        if K < 1:
            raise ValueError('positive K required')
        rows = [self.ranked(u)[:3] for u in range(len(self.schedules))]
        return sorted(product(*rows), key=lambda ss: (-sum(len(s.keep) for s in ss), sum(s.energy for s in ss), tuple(s.keep for s in ss)))[:K]

    def _bounded_schedule(self, u, keep, bounds):
        tb = self.tables; seq = tb.instance['visits'][u]
        previous = np.full(tb.N+1, np.inf)
        previous[tb.ex.takeoff_end] = tb.takeoff_energy(u)
        node = tb.ex.initial_nodes[u]; parents = []; lengths = []
        required = max((j+1 for j, (k, phase) in enumerate(seq) if k in keep and phase == 'a'), default=0)
        best = math.inf, 0, 0
        for pointer in range(len(seq)+1):
            if pointer >= required:
                values = previous[:tb.D+1]+tb.travel_energy(node, tb.ex.final_nodes[u])[tb.D::-1]+tb.landing_energy(u)
                time = int(np.argmin(values))
                if values[time] < best[0]:
                    best = float(values[time]), pointer, time
            if pointer == len(seq):
                break
            k, phase = seq[pointer]; task = tb.tasks[k]; dest = task['z_'+phase]
            costs, parent = advance(previous, tb.travel_energy(node, dest), task['L_'+phase], *bounds[pointer])
            previous = costs+min(p.E for h in tb.lib.heights for p in tb.lib.service(k, phase, h))
            parents.append(parent); lengths.append(task['L_'+phase]); node = dest
        energy, pointer, time = best; starts = []
        if math.isfinite(energy):
            for j in range(pointer-1, -1, -1):
                starts.append(time-lengths[j]); time = int(parents[j][time])
            starts.reverse()
        return Schedule(u, keep, energy, tuple(starts), pointer)

    def reference_schedule(self, schedule):
        u, keep = schedule.uav, schedule.keep; key = u, keep
        if key in self.reference_cache:
            return self.reference_cache[key]
        tic = perf_counter(); tb = self.tables; seq = tb.instance['visits'][u]
        pairs = {k: (seq.index([k, 's']), seq.index([k, 'a'])) for k in keep}
        bounds = []
        for k, phase in seq:
            task = tb.tasks[k]
            lo, hi = tb.windows[k, phase] if k in keep else (0, tb.D-task['L_'+phase])
            if k in keep and phase == 'a':
                lo = max(lo, int(tb.ready_min[k, min(tb.N, tb.windows[k, 's'][0])]))
                hi = min(hi, math.floor(task['d']/tb.p['delta'])-task['L_a'])
            bounds.append((lo, hi))
        heap = []; seen = set(); count = 0
        def push(box):
            nonlocal count
            box = tuple(box)
            if box in seen or any(lo > hi for lo, hi in box):
                return
            seen.add(box)
            candidate = self._bounded_schedule(u, keep, box); count += 1
            if math.isfinite(candidate.energy):
                heappush(heap, (candidate.energy, count, candidate, box))
        push(bounds)
        answer = Schedule(u, keep, math.inf, (), 0)
        while heap:
            _, _, candidate, box = heappop(heap)
            violation = None
            for k, (si, ai) in pairs.items():
                ts, ta = candidate.starts[si], candidate.starts[ai]
                H = math.floor(tb.tasks[k]['H']/tb.p['delta'])
                if ta > ts+H:
                    violation = ('fresh', k, si, ai, ts, ta, H); break
                if ta < tb.ready_min[k, ts]:
                    violation = ('ready', k, si, ai, ts, ta, H); break
            if violation is None:
                answer = candidate; break
            kind, k, si, ai, ts, ta, H = violation
            first, second = list(box), list(box)
            if kind == 'fresh':
                pivot = (ts+ta-H+1)//2
                first[si] = (max(box[si][0], pivot), box[si][1])
                second[si] = (box[si][0], min(box[si][1], pivot-1))
                second[ai] = (box[ai][0], min(box[ai][1], pivot-1+H))
            else:
                latest_s = int(np.searchsorted(tb.ready_min[k], ta, side='right')-1)
                pivot = (latest_s+ts+1)//2
                first[si] = (box[si][0], min(box[si][1], pivot-1))
                second[si] = (max(box[si][0], pivot), box[si][1])
                second[ai] = (max(box[ai][0], int(tb.ready_min[k, pivot])), box[ai][1])
            push(first); push(second)
        self.reference_cache[key] = answer
        self.reference_diagnostics[key] = dict(dp_solves=count, build_s=perf_counter()-tic,
            C2_energy_J=schedule.energy, reference_energy_J=answer.energy,
            gap_J=answer.energy-schedule.energy, exact=True)
        return answer

    def reference_best(self):
        selected = []
        for u in range(len(self.schedules)):
            best = None
            for relaxed in self.ranked(u):
                if best is not None and len(relaxed.keep) < len(best.keep):
                    break
                candidate = self.reference_schedule(relaxed)
                if candidate.energy > self.tables.p['battery']:
                    continue
                if best is None or (-len(candidate.keep), candidate.energy, candidate.keep) < (-len(best.keep), best.energy, best.keep):
                    best = candidate
            if best is None:
                raise ValueError('no battery-feasible freshness-aware reference schedule')
            selected.append(best)
        return tuple(selected)

    def reference_combinations(self, K=16):
        rows = []
        for u in range(len(self.schedules)):
            candidates = [self.reference_schedule(s) for s in self.ranked(u)]
            rows.append(sorted((s for s in candidates if s.energy <= self.tables.p['battery']),
                        key=lambda s: (-len(s.keep), s.energy, s.keep))[:3])
        combinations = sorted(product(*rows), key=lambda ss: (-sum(len(s.keep) for s in ss), sum(s.energy for s in ss), tuple(s.keep for s in ss)))
        from .grammar import Grammar
        grammar = Grammar(self.tables); output = []
        for schedules in combinations:
            c = self._assemble(schedules, grammar_locations=True)
            if grammar.validate(c.tokens(self.tables))['valid']:
                output.append((schedules, c))
                if len(output) == K:
                    break
        return output

    def commitment(self, schedules=None):
        from .grammar import Grammar
        schedules = self.reference_best() if schedules is None else tuple(self.reference_schedule(s) for s in schedules)
        c = self._assemble(schedules, grammar_locations=True)
        validation = Grammar(self.tables).validate(c.tokens(self.tables))
        if not validation['valid']:
            raise ValueError(f'admissible reference failed grammar: {validation}')
        return c

    def literal_commitment(self, schedules=None):
        return self._assemble(self.best() if schedules is None else schedules, grammar_locations=False)

    def _assemble(self, schedules, *, grammar_locations):
        from .grammar import Commitment, TaskCommitment
        tb = self.tables
        layers, used = [], set()
        from lawn_mec.env.geometry import segment_box, Box
        for u, own in enumerate(tb.own):
            def score(ell):
                return sum(not any(segment_box(tb.lib.point(tb.tasks[k]['z_s'], tb.lib.heights[ell]), bs, Box.from_dict(b)) for b in tb.instance['buildings']) for k in own for bs in tb.instance['bs'])
            ell = min((i for i in range(len(tb.lib.heights)) if i not in used), key=lambda i: (-score(i), i))
            layers.append(ell); used.add(ell)
        board = np.zeros((tb.p['M'], tb.N), dtype=bool)
        tasks, slack = [None]*len(tb.tasks), {}
        for u, schedule in enumerate(schedules):
            starts = {tuple(visit): start for visit, start in zip(tb.instance['visits'][u], schedule.starts)}
            pending = []
            for k in tb.own[u]:
                if k not in schedule.keep:
                    tasks[k] = TaskCommitment('G'); continue
                t = tb.tasks[k]; ts, ta = starts[k, 's'], starts[k, 'a']
                bs, ba = bin_of(tb.visit_bins[k, 's'], ts), bin_of(tb.visit_bins[k, 'a'], ta)
                release = ts+t['L_s']
                options = [(release+math.ceil(t['C']/(tb.p['delta']*tb.p['F_local'])), 0, None, ())]
                for m in range(1, tb.p['M']+1):
                    duration = tb.node_upload(k, m, layers[u])
                    slots = tuple(n for n in range(release, tb.N) if not board[m-1, n])[:duration]
                    if len(slots) != duration:
                        continue
                    if grammar_locations:
                        try:
                            rb = bin_of(tb.upload_bins[k], slots[0])
                        except ValueError:
                            continue
                        rlo, rhi = tb.upload_bins[k][rb]
                        if max(rlo, tb.visit_bins[k, 's'][bs][0]+t['L_s']) > min(rhi, tb.visit_bins[k, 'a'][ba][1]):
                            continue
                    cpu = finish(t['C'], tb.cpu_capacity[m-1], slots[-1]+1, tb.N+1)
                    down = math.ceil(t['O']/tb.radio_capacity[m-1, 1])
                    options.append((cpu+down, m, slots[0], slots))
                ready, m, upload, slots = min(options)
                r = None if m == 0 else bin_of(tb.upload_bins[k], upload)
                tasks[k] = TaskCommitment('S', m, bs, ba, r, 'Y')
                slack[k] = t['d']/tb.p['delta']-t['L_a']-ready
                if m:
                    pending.append((m, slots))
            for m, slots in pending:
                board[m-1, list(slots)] = True
        from dataclasses import replace
        ordered = sorted(slack, key=lambda k: (slack[k], k))
        for rank, k in enumerate(ordered):
            tasks[k] = replace(tasks[k], priority='XYZ'[min(2, 3*rank//len(ordered))])
        return Commitment(tuple(layers), tuple(tasks))
