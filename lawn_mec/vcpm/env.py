import numpy as np
from lawn_mec.marl_v2.env import MarlEnv, HeadAction, choice_features, token, ALL_HEADS, HEADS
from .features import own_tokens, intent_tokens, encode_commitment, interval, FactoredObservation

KAPPA_COLUMN = 15
PRIORITY_BANDS = {'X': (12, 16), 'Y': (6, 11), 'Z': (1, 5)}
KAPPA_TIMINGS = ('design', 'fix')
FORCED_HEADS = ('layer', 'location', 'priority', 'timing')
SWITCH_DEFAULTS = dict(kappa_timing='design', force_safe=False, pace='off')
PACE_MODES = ('off', 'band', 'ready', 'repair', 'floor_only', 's_only', 'a_only')


def check_pace(pace, force_safe=False):
    if pace not in PACE_MODES:
        raise ValueError(f'pace must be one of {PACE_MODES}')
    if pace != 'off' and force_safe:
        raise ValueError('pace and force_safe are mutually exclusive')
    return pace


def check_kappa_timing(kappa_timing):
    if kappa_timing not in KAPPA_TIMINGS:
        raise ValueError(f'kappa_timing must be one of {KAPPA_TIMINGS}, got {kappa_timing!r}')
    return kappa_timing


def switch_record(kappa_timing='design', force_safe=False, pace='off'):
    if not isinstance(force_safe, bool):
        raise TypeError('force_safe must be a bool')
    values = dict(kappa_timing=check_kappa_timing(kappa_timing), force_safe=force_safe,
                  pace=check_pace(pace, force_safe))
    return {k: v for k, v in values.items() if v != SWITCH_DEFAULTS[k]}


def recorded_switches(record):
    return {k: record.get(k, default) for k, default in SWITCH_DEFAULTS.items()
            if k != 'pace' or record.get(k, default) != default}


def postponement_kappa(tables, c, k, phase, slot, candidates):
    times = np.rint(candidates[:, 0]*tables.N).astype(int)
    if c.g != 'S':
        return (times == slot).astype(np.float32)
    lo, hi = interval(tables, k, phase, getattr(c, phase))
    if slot < hi:
        return ((slot < times) & (times <= hi)).astype(np.float32)
    return (times == slot).astype(np.float32)


def consistency(tables, commitment, u, slot, head, candidates, request=None, *, kappa_timing='design'):
    check_kappa_timing(kappa_timing)
    result = np.zeros(len(candidates), np.float32)
    if commitment is None or head in ('route', 'offset'):
        return result
    if head == 'layer':
        return (np.rint(candidates[:, 0]*7).astype(int) == commitment.layers[u]).astype(np.float32)
    if head in ('timing', 'bin'):
        visit = None if request is None else request.get('visit')
        if visit is None:
            return result
        k, phase = visit; c = commitment.tasks[k]
        if kappa_timing == 'fix' and head == 'timing' and request.get('reason') == 'deferral':
            return postponement_kappa(tables, c, k, phase, slot, candidates)
        if c.g != 'S':
            return result
        lo, hi = interval(tables, k, phase, getattr(c, phase))
        if head == 'timing':
            times = np.rint(candidates[:, 0]*tables.N).astype(int)
            result[:] = (lo <= times) & (times <= hi)
        else:
            left = np.rint(candidates[:, 1]*tables.N).astype(int)
            right = np.rint(candidates[:, 2]*tables.N).astype(int)
            result[:] = (left <= hi) & (right >= lo)
    elif head in ('location', 'priority'):
        k = int(round(float(candidates[0, 1])*len(tables.tasks)))
        c = commitment.tasks[k]
        if head == 'location' and c.location is not None:
            result[:] = np.rint(candidates[:, 0]*(tables.p['M']+1)).astype(int) == c.location
        elif head == 'priority':
            lo, hi = PRIORITY_BANDS[c.priority]
            levels = np.rint(candidates[:, 0]*16).astype(int)
            stage = int(round(float(candidates[0, 2])*2))
            left, right = interval(tables, k, 'upload', c.upload)
            if stage == 0 and c.upload is not None and left <= slot <= right:
                lo = hi
            result[:] = (lo <= levels) & (levels <= hi)
    return result


def candidate_addresses(tables, head, candidates, request=None):
    if head == 'layer':
        return [('layer', int(round(float(x[0])*7))) for x in candidates]
    if head in ('timing', 'bin', 'offset', 'route'):
        visit = None if request is None else request.get('visit')
        prefix = (head, None if visit is None else tuple(visit))
        if head == 'timing':
            return [(*prefix, int(round(float(x[0])*tables.N))) for x in candidates]
        if head in ('offset', 'bin'):
            return [(*prefix, *(int(round(float(v)*tables.N)) for v in x[1:3 if head == 'bin' else 2])) for x in candidates]
        return [(*prefix, int(round(float(x[0])*4))) for x in candidates]
    if head == 'location':
        return [(head, int(round(float(x[1])*len(tables.tasks))),
                 int(round(float(x[0])*(tables.p['M']+1)))) for x in candidates]
    if head == 'priority':
        return [(head, int(round(float(x[1])*len(tables.tasks))),
                 int(round(float(x[2])*2)), int(round(float(x[0])*16))) for x in candidates]
    raise ValueError(head)


def safe_support(tables, commitment, u, slot, head, candidates, mask, request=None, *, kappa_timing='design'):
    mask = np.asarray(mask, bool)
    if commitment is None or head not in FORCED_HEADS:
        return mask, False, False
    kappa = candidates[:, KAPPA_COLUMN]
    if head == 'timing' and kappa_timing != 'fix':
        kappa = consistency(tables, commitment, u, slot, head, candidates, request, kappa_timing='fix')
    support = mask & (kappa == 1)
    if support.any():
        return support, True, False
    return mask, True, True


def pace_support(env, head, candidates, mask, request, mode):
    mask = np.asarray(mask, bool)
    if mode == 'off' or head != 'timing' or env.plan is None or request is None or request.get('visit') is None:
        return mask, False, False, False
    times = np.rint(candidates[:, 0]*env.N).astype(int)
    k, phase = request['visit']; c = env.plan.tasks[k]
    if mode == 'repair' and phase == 'a' and request.get('reason') in ('initial', 'visit_end', 'deferral'):
        from .ready_repair import repair_support
        repaired = repair_support(env, times, mask, request)
        if repaired is not None:
            return repaired
    if mode == 'repair':
        mode = 'band'
    if request.get('reason') == 'deferral':
        if c.g == 'G' and phase == 'a':
            support = mask & (times == env.executor.state.n)
            return (support, True, False, False) if support.any() else (mask, False, True, False)
        return mask, False, False, False
    if (request.get('reason') not in ('initial', 'visit_end') or c.g != 'S'
            or getattr(c, phase) is None or (mode == 's_only' and phase != 's')
            or (mode == 'a_only' and phase != 'a')):
        return mask, False, False, False
    lo, hi = interval(env.tables, k, phase, getattr(c, phase))
    support = mask & (times >= lo)
    if mode != 'floor_only':
        support &= times <= hi
    override = False
    if phase == 'a' and support.any() and mode in ('band', 'ready'):
        r = env.readiness(request)
        if r is not None and not r.certified:
            band_ok = (support & (times >= r.lo) & (times <= r.limit)).any()
            salvage = mask & (times >= max(r.earliest, r.lo)) & (times <= r.limit)
            if not band_ok and salvage.any():
                support = salvage; override = True
            if mode == 'ready':
                preferred = support & (times >= r.hi) & (times <= r.limit)
                if preferred.any():
                    support = preferred
    if support.any():
        return support, True, False, override
    return mask, False, True, override


def force_safe_summary(counts):
    heads = {h: list(counts[h]) for h in FORCED_HEADS if h in counts}
    return dict(decisions=sum(v[0] for v in heads.values()), fallbacks=sum(v[1] for v in heads.values()), heads=heads)


def merge_force_safe(summaries):
    counts = {}
    for summary in summaries:
        for head, (decisions, fallbacks) in summary['heads'].items():
            row = counts.setdefault(head, [0, 0]); row[0] += decisions; row[1] += fallbacks
    return force_safe_summary(counts)


class CommitmentEnv(MarlEnv):
    def __init__(self, instance, *, tables, public, commitment=None, kappa_timing='design', force_safe=False,
                 pace='off', **kwargs):
        self.tables, self.public, self.plan = tables, public, commitment
        self.kappa_timing, self.force_safe = check_kappa_timing(kappa_timing), bool(force_safe)
        self.pace = check_pace(pace, self.force_safe)
        self.pace_counts = dict(forced=0, bind=0, empty=0, override=0)
        if self.pace == 'repair':
            from .ready_repair import init_counts
            init_counts(self)
        self.force_counts = {}
        super().__init__(instance, commitment=None, **kwargs)
        if self.pace != 'off' and not self.flags.f15:
            raise ValueError('pace requires f15')
        if not self.flags.f4:
            raise ValueError('vcpm requires the F4 fixed critic schema')
        if (self.kappa_timing != 'design' or self.force_safe) and not self.flags.f1:
            raise ValueError('kappa_timing=fix and force_safe act on the F1 hazard timing head')
        self.commitment_vector = encode_commitment(tables, commitment)
        self.boards = [intent_tokens(tables, commitment, u) for u in range(self.U)] if commitment else None

    def local_observation(self, u, request=None, *, with_prefix=True):
        base = super().local_observation(u, request, with_prefix=with_prefix)
        rows = [base]
        if self.plan is not None:
            rows += [own_tokens(self.tables, self.plan, u, self.executor.state.n), self.boards[u]]
        return FactoredObservation(np.concatenate(rows), self.public)

    def critic_state(self):
        state = super().critic_state()
        if self.flags.f14c:
            return np.concatenate([state[:-1], self.commitment_vector, state[-1:]])
        return np.concatenate([state, self.commitment_vector])

    def force_safe_stats(self):
        return force_safe_summary(self.force_counts)

    def pace_stats(self):
        return dict(self.pace_counts)

    def _choose(self, actor, u, head, features, mask, records, request=None):
        mask = np.asarray(mask, bool)
        if self.flags.f17:
            from lawn_mec.marl_v2.guards import guard_support
            mask = guard_support(self, head, features, mask, request)
        if not mask.any():
            raise ValueError(f'empty executor support for {head}')
        observation = self.local_observation(u, request)
        candidates = choice_features(features)
        if np.any(candidates[:, KAPPA_COLUMN]):
            raise ValueError('F13 candidate schema now occupies reserved κ column')
        candidates[:, KAPPA_COLUMN] = consistency(self.tables, self.plan, u,
            self.executor.state.n, head, candidates, request, kappa_timing=self.kappa_timing)
        if self.pace != 'off':
            support, forced, fallback, override = pace_support(self, head, candidates, mask, request, self.pace)
            self.pace_counts['forced'] += int(forced)
            self.pace_counts['empty'] += int(fallback)
            self.pace_counts['override'] += int(override)
            if forced and not np.array_equal(support, mask) and hasattr(actor, 'sample_context'):
                raw, _ = actor.sample_context(u, self.executor.state.n, head, observation, candidates, mask,
                                             candidate_addresses(self.tables, head, candidates, request))
                self.pace_counts['bind'] += int(not support[raw])
            mask = support
        if self.force_safe:
            mask, forced, fallback = safe_support(self.tables, self.plan, u, self.executor.state.n, head,
                                                  candidates, mask, request, kappa_timing=self.kappa_timing)
            if forced:
                row = self.force_counts.setdefault(head, [0, 0]); row[0] += 1; row[1] += int(fallback)
        if hasattr(actor, 'sample_context'):
            action, logp = actor.sample_context(u, self.executor.state.n, head, observation,
                candidates, mask, candidate_addresses(self.tables, head, candidates, request))
        else:
            action, logp = actor.sample(u, head, observation, candidates, mask)
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('actor selected a masked choice')
        if self.pace == 'repair':
            from .ready_repair import record_wait
            record_wait(self, head, request, candidates, action)
        records.append(HeadAction(u, self.executor.state.n, head, observation,
                                  candidates, mask.copy(), action, logp))
        physical = candidates[action].copy(); physical[KAPPA_COLUMN] = 0
        self.prefix[u].append(token(5, [ALL_HEADS.index(head)/len(HEADS), *physical]))
        return action
