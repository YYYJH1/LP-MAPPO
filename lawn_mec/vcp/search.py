from dataclasses import replace
from itertools import zip_longest
import math
import time
import numpy as np
from lawn_mec.exec_v2.search import objective
from lawn_mec.exec_v2.p0_loss import terminal_loss as _p0_terminal_loss
from lawn_mec.exec_v2.policies import rollout as hand_rollout
from .grammar import Commitment, TaskCommitment, Grammar
from .tables import bin_of
from .executor import rollout, fork_rollout


def readiness_counts(audit):
    tasks = audit['tasks']
    offloaded = [t for t in tasks if t['location'] is not None and t['location'] > 0]
    return dict(result_ready_unknown=sum(t['result_ready'] == 'unknown' for t in tasks),
                offloaded_tasks=len(offloaded),
                offloaded_ready_unknown=sum(t['result_ready'] == 'unknown' for t in offloaded))


def terminal_loss(audit, instance, zeta=.85):
    return _p0_terminal_loss(audit, instance, zeta)['loss']


def rule_commitment(instance, tables, *, execution_layer='evaluation'):
    if execution_layer not in ('evaluation', 'audit'):
        raise ValueError('execution_layer must be evaluation or audit')
    result = hand_rollout(instance, library=tables.lib, planner=tables.planner, execution_layer=execution_layer)
    layers = next(d['choices']['layers'] for d in result['decisions'] if d['kind'] == 'layers')
    tasks = []
    for k, task in enumerate(result['audit']['tasks']):
        if not task['completed']:
            tasks.append(TaskCommitment('G')); continue
        location = next(d['choices']['locations'][k] for d in result['decisions'] if d['kind'] == 'locations' and k in d['choices']['locations'])
        upload = next((row['n'] for row in result['logs'] if row['sigma'][k][0] > 0), None) if location else None
        tasks.append(TaskCommitment('S', location, bin_of(tables.visit_bins[k, 's'], task['n_s']),
                     bin_of(tables.visit_bins[k, 'a'], task['n_a']), None if upload is None else bin_of(tables.upload_bins[k], upload), 'Y'))
    ordered = sorted((k for k, c in enumerate(tasks) if c.g == 'S'),
                     key=lambda k: (instance['tasks'][k]['d']/tables.p['delta']-result['audit']['tasks'][k]['n_s']-instance['tasks'][k]['L_s'], k))
    for rank, k in enumerate(ordered):
        tasks[k] = replace(tasks[k], priority='XYZ'[min(2, 3*rank//len(ordered))])
    return Commitment(tuple(layers), tuple(tasks)), result


def moves(commitment, grammar, audit=None):
    tb = grammar.tables
    order = list(range(len(commitment.tasks)))
    if audit is not None:
        order.sort(key=lambda k: (audit['tasks'][k]['completed'], k))
    groups = [[], [], [], []]
    for k in order:
        c = commitment.tasks[k]
        if c.g == 'S':
            groups[0].append((k, 'g', TaskCommitment('G')))
            for m in range(tb.p['M']+1):
                if m == c.location:
                    continue
                for r in ((None,) if m == 0 else ((c.upload,) if c.upload is not None else range(8))):
                    groups[1].append((k, 'location', replace(c, location=m, upload=r)))
            for field in ('s', 'a', 'upload'):
                value = getattr(c, field)
                if value is None:
                    continue
                for b in sorted((i for i in range(8) if i != value), key=lambda i: (abs(i-value), i)):
                    groups[2].append((k, field, replace(c, **{field: b})))
            for priority in 'XYZ':
                if priority != c.priority:
                    groups[3].append((k, 'priority', replace(c, priority=priority)))
        else:
            prefix = commitment.tokens(tb)[:grammar.task_start[k]]
            rows = grammar._rows(prefix, k)
            for m, bs, ba, r in rows:
                from .grammar import LOCATIONS, BINS
                groups[0].append((k, 'g', TaskCommitment('S', LOCATIONS.index(m), BINS.index(bs), BINS.index(ba), None if r == '-' else BINS.index(r), 'Y')))
    for group in zip_longest(*groups):
        for item in group:
            if item is None:
                continue
            k, field, changed = item
            tasks = list(commitment.tasks); tasks[k] = changed
            candidate = replace(commitment, tasks=tuple(tasks))
            yield dict(task=k, field=field, value=changed.labels()), candidate


def local_search(instance, tables, start, *, variant='earliest_feasible', budget=500, seed=0, callback=None,
                 execution_layer='evaluation'):
    if not 1 <= budget <= 500:
        raise ValueError('budget must be 1..500 completed evaluations')
    if execution_layer not in ('evaluation', 'audit'):
        raise ValueError('execution_layer must be evaluation or audit')
    grammar = Grammar(tables); tic = time.process_time()
    validation = grammar.validate(start.tokens(tables))
    if not validation['valid']:
        raise ValueError(f'DP-Commit is outside the canonical grammar: {validation}')
    result, snapshots = rollout(instance, start, tables, variant, capture=True,
                                execution_layer=execution_layer)
    best, best_result = start, result
    key = objective(result['audit'], grammar.c_min)
    trace = [dict(evaluation=1, objective=list(key), accepted=True, move='DP-Commit', fork_slot=0, **readiness_counts(result['audit']))]
    seen = {start}; evaluations = 1; rejects = 0; rng = np.random.default_rng(seed)
    if callback:
        callback(trace[-1], best, best_result)
    while evaluations < budget:
        changed = False
        for move, candidate in moves(best, grammar, best_result['audit']):
            if candidate in seen:
                continue
            seen.add(candidate)
            if not grammar.validate(candidate.tokens(tables))['valid']:
                rejects += 1; continue
            run, trial_snapshots = fork_rollout(instance, candidate, tables, variant, snapshots)
            evaluations += 1
            trial = objective(run['audit'], grammar.c_min); accept = trial < key
            trace.append(dict(evaluation=evaluations, objective=list(trial), accepted=accept, move=move, fork_slot=run['fork_slot'], **readiness_counts(run['audit'])))
            if accept:
                best, best_result, key, snapshots = candidate, run, trial, trial_snapshots
                changed = True
            if callback:
                callback(trace[-1], best, best_result)
            if accept or evaluations >= budget:
                break
        if evaluations >= budget:
            break
        if changed:
            continue
        neighbours = [(m, c) for m, c in moves(best, grammar, best_result['audit']) if grammar.validate(c.tokens(tables))['valid']]
        rng.shuffle(neighbours)
        found = False
        for first_move, first in neighbours:
            for second_move, candidate in moves(first, grammar):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if not grammar.validate(candidate.tokens(tables))['valid']:
                    rejects += 1; continue
                run, trial_snapshots = fork_rollout(instance, candidate, tables, variant, snapshots)
                evaluations += 1; trial = objective(run['audit'], grammar.c_min); accept = trial < key
                trace.append(dict(evaluation=evaluations, objective=list(trial), accepted=accept, move=[first_move, second_move], fork_slot=run['fork_slot'], **readiness_counts(run['audit'])))
                if accept:
                    best, best_result, key, snapshots = candidate, run, trial, trial_snapshots
                if callback:
                    callback(trace[-1], best, best_result)
                found = True
                break
            if found:
                break
        if not found:
            break
    return dict(commitment=best, result=best_result, trace=trace, completed_evaluations=evaluations,
                grammar_rejections=rejects, cpu_s=time.process_time()-tic,
                stop='evaluation_budget' if evaluations == budget else 'structured_neighbourhood_exhausted')
