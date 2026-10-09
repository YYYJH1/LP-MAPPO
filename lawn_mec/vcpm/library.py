from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import numpy as np
from lawn_mec.vcp.grammar import Commitment, TaskCommitment
from lawn_mec.vcp.search import moves


def pack(c):
    return asdict(c)


def unpack(data):
    return Commitment(tuple(data['layers']), tuple(TaskCommitment(**row) for row in data['tasks']))


@dataclass(frozen=True)
class Entry:
    commitment: Commitment
    source: str
    loss: float
    probability: float


class Library:
    def __init__(self, instance, entries, round_index):
        self.instance, self.entries, self.round = str(instance), tuple(entries), round_index
        if not self.entries or any(not math.isfinite(e.loss) or e.probability < 0 for e in self.entries):
            raise ValueError('invalid library entries')
        if not math.isclose(sum(e.probability for e in self.entries), 1., abs_tol=1e-10):
            raise ValueError('library probabilities must sum to one')

    def sample(self, rng):
        return self.entries[int(rng.choice(len(self.entries), p=[e.probability for e in self.entries]))].commitment

    def to_dict(self):
        return dict(version=1, instance=self.instance, round=self.round,
                    entries=[dict(commitment=pack(e.commitment), source=e.source,
                                  loss=e.loss, probability=e.probability) for e in self.entries])

    @classmethod
    def from_dict(cls, data):
        if data['version'] != 1:
            raise ValueError('unknown library schema')
        return cls(data['instance'], [Entry(unpack(e['commitment']), e['source'],
                   e['loss'], e['probability']) for e in data['entries']], data['round'])

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False)+'\n')


def perturb(context, seed, start=None):
    rng = np.random.default_rng(seed); result = start or context.reference
    for _ in range(int(rng.integers(1, 3))):
        choices = []
        for move, candidate in moves(result, context.grammar):
            if move['field'] in ('s', 'a', 'upload'):
                old = getattr(result.tasks[move['task']], move['field'])
                new = getattr(candidate.tasks[move['task']], move['field'])
                if old is not None and new is not None and abs(old-new) > 3:
                    continue
            if context.grammar.validate(candidate.tokens(context.tables))['valid'] and context.dp.precheck(candidate.keep_sets(context.tables))['passed']:
                choices.append(candidate)
        if choices:
            result = choices[int(rng.integers(len(choices)))]
    return result


def build_round(context, instance, mode, samples, old, evaluate, *, seed,
                round_index, rho_new=.6, perturbation=.1):
    if len(samples) != 4 or len(old.entries) < 2:
        raise ValueError('Alg.3 requires four samples and at least two old entries')
    if not 0 <= perturbation <= .1 or not 0 <= rho_new <= 1-perturbation:
        raise ValueError('invalid new/perturbation mass')
    ranked = sorted(enumerate(old.entries), key=lambda ie: ie[0])
    scored = [(evaluate(e.commitment), i, e.commitment) for i, e in ranked]
    best = sorted(scored, key=lambda row: (row[0], row[1]))[:2]
    residual = 1-rho_new-perturbation
    rows = [Entry(c, source, evaluate(c), rho_new/5) for source, c in
            [('mode', mode), *[(f'sample_{i}', c) for i, c in enumerate(samples)]]]
    rows += [Entry(c, 'old_best', loss, residual/4) for loss, _, c in best]
    rows += [Entry(context.reference, 'dp_commit', evaluate(context.reference), residual/2)]
    if perturbation:
        c = perturb(context, seed)
        rows.append(Entry(c, 'perturbation', evaluate(c), perturbation))
    return Library(instance, rows, round_index)


def dp_library(context, instance, evaluate, *, seed, round_index):
    return fixed_library(context, instance, context.reference, 'dp_commit', evaluate,
                         seed=seed, round_index=round_index)


def fixed_library(context, instance, base, source, evaluate, *, seed, round_index):
    p = .25 if round_index == 0 else .1
    altered = perturb(context, seed, start=base)
    return Library(instance, [Entry(base, source, evaluate(base), 1-p),
                             Entry(altered, 'perturbation', evaluate(altered), p)], round_index)


def initial_library(context, instance, evaluate, *, seed, hand_evaluate=None, execution_layer=None):
    if getattr(context, 'toy', False):
        rule = context.reference
    else:
        from lawn_mec.vcp.search import rule_commitment
        rule, _ = rule_commitment(context.instance, context.tables, execution_layer=execution_layer)
    candidates = [c for _, c in context.dp.reference_combinations(16)]
    if not candidates:
        raise ValueError('no EKS candidates')
    if not getattr(context, 'toy', False) and hand_evaluate is None:
        raise ValueError('Alg.1 EKS selection requires P_hand-EL-C evaluations')
    score = evaluate if hand_evaluate is None else hand_evaluate
    scored = [(score(c), i, c) for i, c in enumerate(candidates)]
    loss, _, best = min(scored, key=lambda x: (x[0], x[1]))
    if hand_evaluate is not None:
        loss = evaluate(best)
    altered = perturb(context, seed)
    return Library(instance, [Entry(rule, 'rule_literal', evaluate(rule), .25),
        Entry(context.reference, 'dp_commit', evaluate(context.reference), .25),
        Entry(best, 'eks_best', loss, .25), Entry(altered, 'perturbation', evaluate(altered), .25)], 0)


def search_library(context, instance, evaluate, budget, *, seed, round_index):
    if budget < 1:
        raise ValueError('positive search budget required')
    initial = [context.reference, *[c for _, c in context.dp.reference_combinations(16)]]
    best = context.reference; best_loss = math.inf; seen = set(); rows = []
    neighbourhood = iter(())
    for i in range(budget):
        candidate = initial[i] if i < len(initial) else None
        if candidate is None:
            for _, trial in neighbourhood:
                if trial in seen:
                    continue
                if context.grammar.validate(trial.tokens(context.tables))['valid'] and context.dp.precheck(trial.keep_sets(context.tables))['passed']:
                    candidate = trial; break
            if candidate is None:
                candidate = perturb(context, seed+i, best)
        loss = evaluate(candidate); seen.add(candidate)
        rows.append((loss, i, candidate))
        if loss < best_loss:
            best, best_loss = candidate, loss
            neighbourhood = iter(moves(best, context.grammar))
    selected = sorted(rows, key=lambda x: (x[0], x[1]))[:8]
    return Library(instance, [Entry(c, 'search', loss, 1/len(selected)) for loss, _, c in selected], round_index)
