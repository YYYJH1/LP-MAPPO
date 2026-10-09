import numpy as np
from lawn_mec.env.geometry import Box
from lawn_mec.eval.registry import IDS, FIELDS

MAX_U, MAX_K, MAX_M, MAX_NODE, MAX_BUILDING = 6, 24, 8, 32, 16
FIELD_NAMES = tuple(sorted({f for fs in FIELDS.values() for f in fs}))
BLOCKS = (('global', 32), ('uav', MAX_U*40), ('task', MAX_K*72),
          ('bs', MAX_M*12), ('node', MAX_NODE*7), ('building', MAX_BUILDING*7),
          ('conditions', len(IDS)*3), ('fields', len(FIELD_NAMES)*3),
          ('commitment', MAX_U*41))
SLICES = {}; cursor = 0
for name, size in BLOCKS:
    SLICES[name] = slice(cursor, cursor+size); cursor += size
STATE_DIM = cursor


def _status(value):
    return [float(value['status'] == k) for k in ('pass', 'unknown', 'fail')]


def encode(env):
    ex = env.executor; s = ex.state; p = s.p; a = env.audit_features
    N = s.N; E = p['battery']
    if any(x > maximum for x, maximum in ((env.U, MAX_U), (env.K, MAX_K),
            (p['M'], MAX_M), (len(s.instance['nodes']), MAX_NODE),
            (len(s.instance['buildings']), MAX_BUILDING))):
        raise ValueError('instance exceeds F13 schema v1 maxima')
    out = np.zeros(STATE_DIM, np.float32)
    def put(name, index, width, values):
        if len(values) > width:
            raise ValueError(f'{name} row overflow: {len(values)} > {width}')
        start = SLICES[name].start+index*width
        out[start:start+len(values)] = values
    put('global', 0, 32, [s.n/N, env.U/MAX_U, env.K/MAX_K, p['M']/MAX_M,
        N/400, p['delta'], p['battery']/1e5, p['F_local']/1e10, p['G_total']/1e10,
        p['bandwidth']/1e7, p['P_ul'], p['P_dl'], p['V_h']/20, p['V_up']/20,
        p['V_down']/20, p['a_h']/10, p['a_z']/10, p['H']/p['T'] if 'H' in p else 0,
        p['kappa']/1e-27, p['beta_L']/1e-6, p['beta_N']/1e-6, p['N0']/1e-20,
        p['alpha_L']/5, p['alpha_N']/5, p['eta_tx'], p['eta_flight'],
        p['D_h']/100, p['D_z']/100, p['V_service']/20, ex.takeoff_end/N,
        ex.planner.landing_start/N, float(getattr(ex, 'service', None) is not None)])
    for u in range(env.U):
        c = ex.commitments[u]
        target = ex.final_nodes[u] if c and c['endpoint'] else (
            s.tasks[s.instance['visits'][u][c['pointer']][0]][
                'z_'+s.instance['visits'][u][c['pointer']][1]] if c else None)
        node = s.route_node[u]
        energies = [v for part in a.energies[u] for v in (part.lo/E, part.hi/E)]
        put('uav', u, 40, [1, *s.q[u]/1000, *s.v[u]/20,
            *np.asarray(s.instance['qI'][u])/1000, *np.asarray(s.instance['qF'][u])/1000,
            s.visit_pointer[u]/(2*MAX_K), s.arrival[u]/N, float(s.arrival_flags[u]),
            -1 if ex.layers is None else ex.lib.heights[ex.layers[u]]/100,
            float(u in ex.pending), float(u in ex.deferrals), *energies,
            1-sum(x.hi for x in a.energies[u])/E, 1-sum(x.lo for x in a.energies[u])/E,
            *([0]*3 if node is None else env._centres[node]/1000),
            float(c is not None), -1 if c is None else c['T']/N,
            -1 if c is None else c['end']/N,
            *([0]*3 if target is None else env._centres[target]/1000)])
        put('commitment', u, 41, [float(env.commitment is not None), *env.commitment_features(u)])
    task_states = env.task_states(critic=True)
    for k, (t, st) in enumerate(zip(s.tasks, task_states)):
        events = [v for name in ('s','g','ul','c','dl','r','a') for v in
                  (float(st.events[name].available), -1 if st.events[name].value is None else st.events[name].value/N)]
        service = [float(a.total[k][bound][i])/t[key] for bound in range(2)
                   for i, key in enumerate(('D','C','O'))]
        ends = [-1 if x is None else x/N for bound in range(2) for x in a.ends[k][bound]]
        loc = ex.locations[k]
        put('task', k, 72, [1, *[float(t['u'] == u) for u in range(MAX_U)],
            *[float(loc == m) for m in range(MAX_M+1)], float(loc is None),
            t['D']/1e8, t['C']/1e10, t['O']/1e7, t['L_s']/N, t['L_a']/N,
            t['d']/p['T'], t['H']/p['T'],
            *env._centres[t['z_s']]/1000, *env._centres[t['z_a']]/1000,
            st.J['s']/t['L_s'], st.J['a']/t['L_a'],
            *[st.W[i]/t[key] for i, key in enumerate(('D','C','O'))],
            *events, *service, *[max(0, 1-x) for x in service[:3]], *ends])
    for m, bs in enumerate(env._bs):
        times = np.linspace(min(s.n,N-1), N-1, 6, dtype=int)
        put('bs', m, 12, [1, *bs/1000, *np.asarray(s.instance['load'][m])[times]/p['G_total'],
            sum(float(st.W[1]) for st in task_states if st.location == m+1)/1e10])
    for key, block in (('nodes','node'), ('buildings','building')):
        for i, value in enumerate(s.instance[key]):
            box = Box.from_dict(value)
            put(block, i, 7, [1, *np.asarray(box.lo)/1000, *np.asarray(box.hi)/1000])
    for i, key in enumerate(IDS):
        value = a.fields['viol_visit_continue'] if key == 'C07' else a.conditions[key]
        put('conditions', i, 3, _status(value))
    for i, key in enumerate(FIELD_NAMES):
        put('fields', i, 3, _status(a.fields[key]))
    if not np.isfinite(out).all():
        raise ValueError('nonfinite physical state')
    return out
