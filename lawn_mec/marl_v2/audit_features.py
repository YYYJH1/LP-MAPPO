from fractions import Fraction
import numpy as np
from lawn_mec.env.checker import check_slot, slot_integrals
from lawn_mec.env.interval import iv
from lawn_mec.eval.registry import IDS, FIELDS, merge, verdict


class AuditFeatures:
    def __init__(self, executor):
        self.ex = executor
        self.n = 0
        K = len(executor.state.tasks)
        self.total = [[[Fraction(0) for _ in range(3)] for _ in range(2)] for _ in range(K)]
        self.ends = [[[None]*3 for _ in range(2)] for _ in range(K)]
        self.energies = [[iv(0) for _ in range(4)] for _ in range(executor.state.p['U'])]
        self.conditions = {k: verdict(0) for k in IDS}
        self.fields = {f: verdict(0) for fs in FIELDS.values() for f in fs}

    def add(self, slot):
        s = self.ex.state
        if slot['n'] != self.n:
            raise ValueError('audit statistics require consecutive slots')
        if s.layer == 'audit' and hasattr(s, 'physics'):
            if s.history is None or s.history.slot['n'] != self.n:
                raise ValueError('audit feature reuse requires the latest executed slot')
            caps, energy = s.physics.integrals(slot, 'audit')
            checked = s.history.checked
        else:
            caps, energy = slot_integrals(s.instance, slot, 'audit', s.cfg)
            checked = check_slot(s.instance, slot, 'audit', s.cfg)
        for k, (t, st) in enumerate(zip(s.tasks, s.states)):
            location = slot['locations'][k]
            stages = (1,) if location == 0 else (0, 1, 2)
            for bound in range(2):
                old_ends = self.ends[k][bound].copy()
                for pos, stage in enumerate(stages):
                    release = st.events['g'].value if pos == 0 else old_ends[stages[pos-1]]
                    if location is None or release is None or release > self.n or old_ends[stage] is not None:
                        continue
                    cap = (Fraction(s.p['delta'])*Fraction(slot['f'][k]) if stage == 1 else
                           Fraction(caps[k][stage].lo if bound == 0 else caps[k][stage].hi))
                    self.total[k][bound][stage] += cap
                    if self.total[k][bound][stage] >= t[('D', 'C', 'O')[stage]]:
                        self.ends[k][bound][stage] = self.n+1
        for u, parts in enumerate(energy):
            for j, part in enumerate(parts):
                self.energies[u][j] += part
        for key, value in checked['conditions'].items():
            self.conditions[key] = merge(self.conditions[key], value)
        for key in ('C01','C02','C03','C07','C09','C10','C20'):
            self.conditions[key] = merge(self.conditions[key], s.conditions[key])
        for key, value in checked['field_conditions'].items():
            self.fields[key] = merge(self.fields[key], value)
        self.fields['viol_visit_continue'] = merge(
            self.fields['viol_visit_continue'], s.fields['viol_visit_continue'])
        self.n += 1

    def ready(self, k, conservative=True):
        st = self.ex.state.states[k]
        if st.location is None:
            return None
        return self.ends[k][0 if conservative else 1][1 if st.location == 0 else 2]


def deadline_failures(ex, audit):
    s = ex.state; dt = s.p['delta']
    failed = np.zeros(len(s.tasks), bool)
    for k, (t, st) in enumerate(zip(s.tasks, s.states)):
        na, ns = st.events['a'].value, st.events['s'].value
        if na is not None:
            ready = audit.ready(k)
            failed[k] = ((na+t['L_a'])*dt > t['d'] or ns is None or
                         (na-ns)*dt > t['H'] or ready is None or ready > na)
    for u in range(s.p['U']):
        pointer = s.visit_pointer[u]
        seq = s.instance['visits'][u]
        if pointer >= len(seq):
            continue
        layers = range(len(ex.lib.heights)) if ex.layers is None else [ex.layers[u]]
        earliest = {}
        for ell in layers:
            node = ex.initial_nodes[u] if s.n == 0 else s.route_node[u]
            time = max(s.n, ex.takeoff_end) if s.n == 0 else s.n
            c = ex.commitments[u]
            for j in range(pointer, len(seq)):
                k, phase = seq[j]; t, st = s.tasks[k], s.states[k]
                dest = t['z_'+phase]
                if j == pointer and c is not None and c['pointer'] == pointer and not c['endpoint']:
                    start = max(c['T'], st.events[phase].value or c['T'])
                else:
                    start = time+int(ex.planner.minimum_times[ell, node, dest])
                if phase == 'a':
                    earliest[k] = min(earliest.get(k, float('inf')), start)
                time = start+t['L_'+phase]; node = dest
        for k, start in earliest.items():
            t, st = s.tasks[k], s.states[k]; ns = st.events['s'].value
            failed[k] |= ((start+t['L_a'])*dt > t['d'] or
                          (ns is not None and (start-ns)*dt > t['H']))
    return failed
