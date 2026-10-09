from dataclasses import dataclass
import numpy as np
from lawn_mec.env.checker import check_slot, slot_integrals
from lawn_mec.env.geometry import Box
from lawn_mec.env.interval import iv
from lawn_mec.env.model import motion
from lawn_mec.env.queue import inject, serve, advance_visit
from lawn_mec.eval.registry import IDS, verdict, merge
from .state import History, freeze


class InfeasibleAction(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(str(details))


@dataclass
class SlotActions:
    edge: object
    accel: object
    visit_start: object
    locations: object
    sigma: object
    power: object
    f: object
    terminal_locations: object = None

    @classmethod
    def from_record(cls, state, record):
        starts = [False]*state.p['U']
        for k, phase in record['active']:
            if state.states[k].J[phase] == 0:
                starts[state.tasks[k]['u']] = True
        return cls(*(record[x] for x in ('edge', 'accel')), starts,
                   *(record[x] for x in ('locations', 'sigma', 'power', 'f')),
                   record.get('terminal_locations'))


def step_v2(state, actions, *, on_infeasible='raise'):
    if on_infeasible not in ('raise', 'record'):
        raise ValueError('on_infeasible must be raise or record')
    if state.n >= state.N:
        raise RuntimeError('episode already finished')
    if isinstance(actions, dict):
        actions = SlotActions(**actions)
    U, K = state.p['U'], len(state.tasks)
    for name, shape in (('accel', (U, 3)), ('sigma', (K, 2)), ('power', (K,)), ('f', (K,))):
        array = np.asarray(getattr(actions, name), float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f'{name} requires finite shape {shape}')
    if len(actions.edge) != U or any(isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer))
                                    or not 0 <= e < len(state.instance['edges']) for e in actions.edge):
        raise ValueError('invalid explicit edge indices')
    if len(actions.visit_start) != U or any(x not in (False, True) for x in actions.visit_start):
        raise ValueError('one binary visit-start flag per UAV required')
    if len(actions.locations) != K:
        raise ValueError('one explicit location per task required')
    for m in actions.locations:
        if m is not None and (isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or not 0 <= m <= state.p['M']):
            raise ValueError('invalid execution location')
    work = state.clone()
    n, dt, eps = work.n, work.p['delta'], work.cfg['epsilon_num']
    local = {key: verdict(0) for key in IDS}
    for u, chosen in enumerate(actions.edge):
        edge = work.instance['edges'][chosen]
        if n and work.edge[u] is not None:
            previous = work.instance['edges'][work.edge[u]]
            work.arrival_flags[u] = n > work.arrival[u] and Box.from_dict(work.instance['nodes'][previous['target']]).contains(work.q[u], eps)
            if work.arrival_flags[u]:
                work.route_node[u], work.arrival[u] = previous['target'], n
        invalid = edge['source'] != work.route_node[u] if work.arrival_flags[u] else chosen != work.edge[u]
        local['C01'] = merge(local['C01'], verdict(float(invalid)))
        work.edge[u] = int(chosen)
    for k, st in enumerate(work.states):
        m = actions.locations[k]
        if st.events['g'].value == n:
            local['C20'] = merge(local['C20'], verdict(float(m is None)))
            st.location = m
        else:
            local['C20'] = merge(local['C20'], verdict(float(m != st.location)))
        if st.location is not None:
            inject(work.tasks[k], st, n)
    active = []
    for u, start in enumerate(actions.visit_start):
        idx, seq = work.visit_pointer[u], work.instance['visits'][u]
        if idx >= len(seq):
            if start:
                raise InfeasibleAction({'visit_start': f'UAV {u} has no remaining visit'})
            continue
        k, phase = seq[idx]
        continuing = 0 < work.states[k].J[phase] < work.tasks[k]['L_'+phase]
        if start and continuing:
            raise InfeasibleAction({'visit_start': f'UAV {u} visit already started'})
        if start and n+work.tasks[k]['L_'+phase] > work.N:
            raise InfeasibleAction({'visit_start': 'insufficient slots for continuous service'})
        if start or continuing:
            active.append((k, phase))
    slot = dict(n=n, q=work.q.tolist(), v=work.v.tolist(),
                edge=[int(e) for e in actions.edge], accel=np.asarray(actions.accel, float).tolist(),
                active=[list(x) for x in active], locations=list(actions.locations),
                W=[st.W.tolist() for st in work.states], sigma=np.asarray(actions.sigma, float).tolist(),
                power=np.asarray(actions.power, float).tolist(), f=np.asarray(actions.f, float).tolist())
    checked = check_slot(work.instance, slot, work.layer, work.cfg)
    for key in IDS:
        local[key] = merge(local[key], checked['conditions'][key])
    bad = {k: v for k, v in local.items() if v['status'] != 'pass'}
    if bad and on_infeasible == 'raise':
        raise InfeasibleAction(bad)
    guard_mu, guard_parts = work.physics.integrals(slot, 'guard')
    mu, parts = work.physics.integrals(slot, work.layer)
    slot['mu'] = np.asarray(guard_mu).tolist()
    slot['energy_parts'] = np.asarray(guard_parts).tolist()
    for k, phase in active:
        advance_visit(work.states[k], phase, work.tasks[k]['L_'+phase], n)
        if work.states[k].J[phase] == work.tasks[k]['L_'+phase]:
            work.visit_pointer[work.tasks[k]['u']] += 1
    for k, st in enumerate(work.states):
        serve(st, guard_mu[k], n)
    for u in range(U):
        for j in range(4):
            work.energies[u][j] += parts[u][j]
        work.B[u] = work.p['battery']-sum(work.energies[u], iv(0))
    work.q, work.v = motion(work.q, work.v, np.asarray(actions.accel, float), dt)
    work.n += 1
    if work.n == work.N:
        terminal = actions.terminal_locations
        if terminal is None:
            terminal = [st.location for st in work.states]
        if len(terminal) != K:
            raise ValueError('one terminal location per task required')
        for k, st in enumerate(work.states):
            if st.events['g'].value == work.N and st.location is None:
                if terminal[k] is None or not isinstance(terminal[k], (int, np.integer)) or isinstance(terminal[k], (bool, np.bool_)) or not 0 <= terminal[k] <= work.p['M']:
                    raise InfeasibleAction({'terminal_location': f'task {k} needs explicit location at N'})
                st.location = int(terminal[k])
            elif terminal[k] != st.location:
                raise InfeasibleAction({'terminal_location': f'task {k} cannot change location'})
            if st.location is not None:
                inject(work.tasks[k], st, work.N)
        slot['terminal_locations'] = list(terminal)
        local['C10'] = merge(local['C10'], verdict(float(np.max(np.linalg.norm(
            work.q-np.asarray(work.instance['qF']), axis=1))), eps))
        for u, edge in enumerate(work.edge):
            target = work.instance['edges'][edge]['target']
            local['C03'] = merge(local['C03'], verdict(float(not Box.from_dict(work.instance['nodes'][target]).contains(work.q[u], eps))))
            B = work.B[u]
            local['C19'] = merge(local['C19'], dict(status='pass' if B.lo >= 0 else 'fail' if B.hi < 0 else 'unknown', residual=-B.lo))
        bad = {k: v for k, v in local.items() if v['status'] != 'pass'}
        if bad and on_infeasible == 'raise':
            raise InfeasibleAction(bad)
    slot['exec_v2_status'] = 'infeasible' if bad else 'ok'
    slot['exec_v2_conditions'] = bad
    for key in IDS:
        work.conditions[key] = merge(work.conditions[key], local[key])
    for key, value in checked['field_conditions'].items():
        work.fields[key] = merge(work.fields[key], value)
    work.diagnostics['rej_corridor_inner'] += checked['rej_corridor_inner']
    work.history = History(work.history, freeze(slot), tuple(tuple(iv(x) for x in row) for row in mu), freeze(checked))
    state.__dict__.update(work.__dict__)
    return slot
