from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
import numpy as np

from lawn_mec.env.geometry import Box
from lawn_mec.env.generator import accept
from lawn_mec.env.interval import iv
from lawn_mec.env.params import checker_config
from lawn_mec.env.queue import TaskState, Event
from lawn_mec.env.checker import _completion
from lawn_mec.eval.registry import IDS, FIELDS, verdict, merge


def freeze(value):
    if isinstance(value, np.ndarray):
        return freeze(value.tolist())
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class History:
    previous: 'History | None'
    slot: object
    capacities: tuple
    checked: object


def clone_task(st):
    return TaskState(st.location, st.W.copy(), st.J.copy(),
                     {k: Event(e.available, e.value) for k, e in st.events.items()},
                     st.injected.copy())


class SimState:
    def __init__(self, instance, *, layer='evaluation', cfg=None):
        if layer not in ('evaluation', 'audit'):
            raise ValueError('state layer must be evaluation or audit')
        inst = dict(instance)
        inst['_geometry_accepted'] = accept(instance)['passed']
        self.instance = freeze(inst)
        self.cfg = freeze(cfg or checker_config())
        from .physics import context
        self.physics = context(self.instance, self.cfg)
        self.layer = layer
        self.p, self.tasks = self.instance['params'], self.instance['tasks']
        self.N = round(self.p['T']/self.p['delta'])
        self.n = 0
        self.q = np.array(inst['qI'], float)
        self.v = np.array(inst['v0'], float)
        self.states = [TaskState() for _ in self.tasks]
        self.route_node = [next((i for i, b in enumerate(inst['nodes'])
                                if Box.from_dict(b).contains(q)), None) for q in self.q]
        self.edge = [None]*self.p['U']
        self.arrival = [0]*self.p['U']
        self.arrival_flags = [True]*self.p['U']
        self.visit_pointer = [0]*self.p['U']
        self.energies = [[iv(0) for _ in range(4)] for _ in range(self.p['U'])]
        self.B = [iv(self.p['battery']) for _ in range(self.p['U'])]
        self.conditions = {k: verdict(0) for k in IDS}
        self.conditions['C02'] = verdict(float(any(z is None for z in self.route_node)))
        self.fields = {f: verdict(0) for fs in FIELDS.values() for f in fs}
        self.diagnostics = {'rej_corridor_inner': 0, 'rej_service_inner': 0}
        self.history = None

    def clone(self):
        other = copy(self)
        other.q, other.v = self.q.copy(), self.v.copy()
        other.states = [clone_task(st) for st in self.states]
        for name in ('route_node', 'edge', 'arrival', 'arrival_flags', 'visit_pointer', 'B'):
            setattr(other, name, getattr(self, name).copy())
        other.energies = [row.copy() for row in self.energies]
        other.conditions = {k: v.copy() for k, v in self.conditions.items()}
        other.fields = {k: v.copy() for k, v in self.fields.items()}
        other.diagnostics = self.diagnostics.copy()
        return other

    def records(self):
        out, node = [], self.history
        while node is not None:
            out.append(node)
            node = node.previous
        return list(reversed(out))

    def logs(self):
        return [thaw(row.slot) for row in self.records()]

    def result(self):
        cs = {k: v.copy() for k, v in self.conditions.items()}
        fields = {k: v.copy() for k, v in self.fields.items()}
        eps, dt = self.cfg['epsilon_num'], self.p['delta']
        rows = self.records()
        cs['C03'] = merge(cs['C03'], verdict(float(self.n != self.N)))
        residuals = np.linalg.norm(self.q-np.array(self.instance['qF']), axis=1)
        cs['C10'] = merge(cs['C10'], verdict(float(max(residuals)), eps))
        for u, edge in enumerate(self.edge):
            if edge is None or not Box.from_dict(self.instance['nodes'][self.instance['edges'][edge]['target']]).contains(self.q[u], eps):
                cs['C03'] = merge(cs['C03'], verdict(1))
        tasks = []
        for k, (t, st) in enumerate(zip(self.tasks, self.states)):
            ns, ng, na = (st.events[x].value for x in ('s', 'g', 'a'))
            lower = upper = ng
            for stage in ((1,) if st.location == 0 else (0, 1, 2)):
                series = [row.capacities[k][stage] for row in rows]
                if stage == 1 and self.layer == 'audit':
                    series = [Fraction(dt)*Fraction(row.slot['f'][k]) for row in rows]
                work = (t['D'], t['C'], t['O'])[stage]
                lower = _completion(series, lower, work, True, self.layer != 'audit')
                upper = _completion(series, upper, work, False, self.layer != 'audit')
            ready = deadline = fresh = None
            if na is not None:
                ready = ('pass' if upper is not None and upper <= na else
                         'fail' if lower is None or lower > na else 'unknown')
                deadline = 'pass' if (na+t['L_a'])*dt <= t['d'] else 'fail'
                fresh = 'pass' if ns is not None and (na-ns)*dt <= t['H'] else 'fail'
            window = [ready, deadline, fresh]
            statuses = [{'status': x or 'fail', 'residual': 0. if x in ('pass', 'unknown') else 1.} for x in window]
            cs['C18'] = merge(cs['C18'], *statuses)
            done = all(st.J[x] == t['L_'+x] for x in ('s', 'a'))
            fields['viol_visit_complete'] = merge(fields['viol_visit_complete'], verdict(float(not done)))
            cs['C07'] = merge(cs['C07'], fields['viol_visit_complete'])
            for name, status in zip(('viol_result_ready', 'viol_deadline', 'viol_freshness'), statuses):
                fields[name] = merge(fields[name], status)
            events = {x: e.as_dict() for x, e in st.events.items()}
            nr = lower if lower == upper else None
            events['r'] = {'available': nr is not None, 'value': nr}
            tasks.append(dict(k=k, u=t['u'], started=ns is not None, completed=done and window == ['pass']*3,
                              events=events, n_s=ns, n_g=ng, n_r=nr, n_a=na, n_r_interval=[lower, upper],
                              location=st.location, J_s=st.J['s'], J_a=st.J['a'], result_ready=ready,
                              deadline=deadline, freshness=fresh,
                              deadline_slack_s=None if na is None else t['d']-(na+t['L_a'])*dt,
                              freshness_slack_s=None if na is None or ns is None else t['H']-(na-ns)*dt))
        uavs = []
        for u, parts in enumerate(self.energies):
            total = sum(parts, iv(0)); B = self.p['battery']-total
            status = 'pass' if B.lo >= 0 else 'fail' if B.hi < 0 else 'unknown'
            cs['C19'] = merge(cs['C19'], {'status': status, 'residual': -B.lo})
            uavs.append(dict(u=u, E_u=(total.lo+total.hi)/2, energy_interval=total.as_list(),
                             **{f'E_{name}': (x.lo+x.hi)/2 for name, x in zip('fcrp', parts)},
                             battery_initial=self.p['battery'], B_N=(B.lo+B.hi)/2,
                             battery_interval=B.as_list(), endpoint_residual=float(residuals[u]),
                             unknown_count=sum(row.checked['conditions'][key]['status'] == 'unknown'
                                               for row in rows for key in ('C13', 'C14'))))
        for cid, names in FIELDS.items():
            if cid not in ('C07', 'C15', 'C17', 'C18'):
                for name in names:
                    fields[name] = merge(fields[name], cs[cid])
        return dict(layer=self.layer, conditions=cs, field_conditions=fields,
                    slots=[thaw(row.checked) for row in rows], tasks=tasks, uavs=uavs,
                    diagnostics=self.diagnostics.copy(), C=sum(t['completed'] for t in tasks))
