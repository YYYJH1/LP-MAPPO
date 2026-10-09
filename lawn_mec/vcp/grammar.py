from dataclasses import dataclass
import math
import numpy as np

BINS = 'ABCDEFGH'
LAYERS = 'ABCDEFG'
LOCATIONS = 'LPQR'
FIELDS = ('g', 'location', 's', 'a', 'upload', 'priority')


@dataclass(frozen=True)
class TaskCommitment:
    g: str
    location: int | None = None
    s: int | None = None
    a: int | None = None
    upload: int | None = None
    priority: str = 'Z'

    def labels(self):
        return (self.g, '-' if self.location is None else LOCATIONS[self.location],
                '-' if self.s is None else BINS[self.s], '-' if self.a is None else BINS[self.a],
                '-' if self.upload is None else BINS[self.upload], self.priority)


@dataclass(frozen=True)
class Commitment:
    layers: tuple
    tasks: tuple

    def tokens(self, tables):
        return tuple(token for u, own in enumerate(tables.own) for token in
                     (LAYERS[self.layers[u]], *(v for k in own for v in self.tasks[k].labels())))

    def keep_sets(self, tables):
        return tuple(tuple(k for k in own if self.tasks[k].g == 'S') for own in tables.own)


class Grammar:
    def __init__(self, tables, zeta=.85):
        self.tables = tables
        if tables.p['M'] > 3 or tables.p['U'] > 7 or not 0 < zeta <= 1:
            raise ValueError('canonical grammar requires M<=3, U<=7, 0<zeta<=1')
        self.c_min = math.ceil(zeta*len(tables.tasks)-1e-9)
        fields = []
        self.task_start = {}
        for u, own in enumerate(tables.own):
            fields.append((u, None, 'layer'))
            for k in own:
                self.task_start[k] = len(fields)
                fields.extend((u, k, name) for name in FIELDS)
        self.fields = tuple(fields)
        self._row_cache = {}

    def _known_bins(self, prefix, current, bs, ba):
        known = {(current, 's'): bs, (current, 'a'): ba}
        for k, at in self.task_start.items():
            if at+4 < len(prefix) and prefix[at] == 'S':
                known[k, 's'] = BINS.index(prefix[at+2])
                known[k, 'a'] = BINS.index(prefix[at+3])
        return known

    def _rows(self, before, k):
        key = before, k
        if key in self._row_cache:
            return self._row_cache[key]
        tb = self.tables; task = tb.tasks[k]; rows = []
        for bs, (slo, shi) in enumerate(tb.visit_bins[k, 's']):
            if slo > shi or not 0 <= slo <= tb.N:
                continue
            for ba, (alo, ahi) in enumerate(tb.visit_bins[k, 'a']):
                if alo > ahi:
                    continue
                lo = int(tb.ready_min[k, slo])
                hi = min(math.floor(task['d']/tb.p['delta'])-task['L_a'], shi+math.floor(task['H']/tb.p['delta']))
                if max(alo, lo) > min(ahi, hi):
                    continue
                known = self._known_bins(before, k, bs, ba); prev = None; valid = True
                for visit in tb.instance['visits'][task['u']]:
                    visit = tuple(visit)
                    if visit not in known:
                        continue
                    a, phase = visit; interval = tb.visit_bins[visit][known[visit]]
                    if prev is not None:
                        b, ph, left = prev
                        bound = left+tb.tasks[b]['L_'+ph]+int(tb.min_time[tb.tasks[b]['z_'+ph], tb.tasks[a]['z_'+phase]])
                        if interval[1] < bound:
                            valid = False; break
                    prev = a, phase, interval[0]
                if not valid:
                    continue
                rows.append(('L', BINS[bs], BINS[ba], '-'))
                for m in range(1, tb.p['M']+1):
                    for r, (rlo, rhi) in enumerate(tb.upload_bins[k]):
                        if max(rlo, slo+task['L_s']) <= min(rhi, ahi):
                            rows.append((LOCATIONS[m], BINS[bs], BINS[ba], BINS[r]))
        if len(self._row_cache) >= 1024:
            self._row_cache.pop(next(iter(self._row_cache)))
        self._row_cache[key] = tuple(rows)
        return self._row_cache[key]

    def legal_options(self, prefix=()):
        prefix = tuple(prefix)
        if len(prefix) >= len(self.fields):
            if len(prefix) == len(self.fields):
                return ()
            raise ValueError('prefix longer than grammar')
        u, k, field = self.fields[len(prefix)]
        if field == 'layer':
            chosen = {prefix[i] for i, (_, _, f) in enumerate(self.fields[:len(prefix)]) if f == 'layer'}
            return tuple(x for x in LAYERS if x not in chosen)
        at = self.task_start[k]; local = prefix[at:]
        if field == 'g':
            has_s = bool(self._rows(prefix, k))
            num_g = sum(prefix[j] == 'G' for j in self.task_start.values() if j < len(prefix))
            return tuple((['S'] if has_s else [])+(['G'] if num_g < len(self.tables.tasks)-self.c_min or not has_s else []))
        if local[0] == 'G':
            return ('Z',) if field == 'priority' else ('-',)
        if field == 'priority':
            return tuple('XYZ')
        rows = self._rows(prefix[:at], k)
        selected = local[1:]
        choices = {r[len(selected)] for r in rows if r[:len(selected)] == selected}
        order = LOCATIONS if field == 'location' else '-'+BINS
        return tuple(x for x in order if x in choices)

    def validate(self, tokens):
        prefix = []
        for index, token in enumerate(tokens):
            options = self.legal_options(prefix)
            if token not in options:
                return dict(valid=False, index=index, field=self.fields[index] if index < len(self.fields) else None, token=token, options=options)
            prefix.append(token)
        return dict(valid=len(prefix) == len(self.fields), index=len(prefix))

    def decode(self, tokens, *, validate=True):
        tokens = tuple(tokens)
        if validate and not self.validate(tokens)['valid']:
            raise ValueError(self.validate(tokens))
        if len(tokens) != len(self.fields):
            raise ValueError('incomplete commitment')
        layers, tasks = [], [None]*len(self.tables.tasks)
        for index, (_, k, field) in enumerate(self.fields):
            if field == 'layer':
                layers.append(LAYERS.index(tokens[index]))
            elif field == 'g':
                g, m, bs, ba, r, priority = tokens[index:index+6]
                tasks[k] = TaskCommitment(g, None if m == '-' else LOCATIONS.index(m), None if bs == '-' else BINS.index(bs), None if ba == '-' else BINS.index(ba), None if r == '-' else BINS.index(r), priority)
        return Commitment(tuple(layers), tuple(tasks))

    def sample(self, seed):
        rng = np.random.default_rng(seed); prefix = []
        while len(prefix) < len(self.fields):
            options = self.legal_options(prefix)
            if not options:
                raise AssertionError('legal prefix has no continuation')
            prefix.append(options[int(rng.integers(len(options)))])
        return self.decode(prefix)

    def complete(self, prefix=()):
        prefix = list(prefix)
        while len(prefix) < len(self.fields):
            prefix.append(self.legal_options(prefix)[0])
        return self.decode(prefix)
