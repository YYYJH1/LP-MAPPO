import json
import math
import numpy as np
from .features import static_features


def commitment_text(commitment, tables):
    from .grammar import LAYERS
    lines = []
    for u, own in enumerate(tables.own):
        lines.append(f'U{u+1} layer {LAYERS[commitment.layers[u]]}')
        lines.extend(f'T{k+1} '+ ' '.join(commitment.tasks[k].labels()) for k in own)
    return '\n'.join(lines)


def tensor_serialization(tables, dp, reference):
    result = static_features(tables, dp, reference)
    p, inst = tables.p, tables.instance
    parameter_names = sorted(k for k, v in p.items() if isinstance(v, (int, float, bool)))
    task_names = ('u', 'z_s', 'z_a', 'L_s', 'L_a', 'D', 'C', 'O', 'd', 'H', 'V_s', 'V_a', 'P_s', 'P_a')
    task_names = tuple(k for k in task_names if all(k in t for t in tables.tasks))
    result.update(parameters=np.asarray([p[k] for k in parameter_names], float),
                  tasks=np.asarray([[t[k] for k in task_names] for t in tables.tasks], float),
                  load=np.asarray(inst['load'], float), bs=np.asarray(inst['bs'], float),
                  starts=np.asarray(inst['qI']), ends=np.asarray(inst['qF']), velocity=np.asarray(inst['v0']),
                  nodes=np.asarray([[b['lo'], b['hi']] for b in inst['nodes']]),
                  buildings=np.asarray([[b['lo'], b['hi']] for b in inst['buildings']]),
                  edges=np.asarray([[e['source'], e['target']] for e in inst['edges']]),
                  corridors=np.asarray([[e['box']['lo'], e['box']['hi']] for e in inst['edges']]),
                  service_boxes=np.asarray([[[t['S_'+phase]['lo'], t['S_'+phase]['hi']] for phase in ('s', 'a')] for t in tables.tasks]),
                  visits_order=np.asarray([[u, k, int(phase == 'a')] for u, seq in enumerate(inst['visits']) for k, phase in seq]))
    names = dict(parameters=parameter_names, tasks=list(task_names), reference_tasks=['keep', 'place', 'first', 'second', 'upload', 'priority'])
    return dict(arrays=result, columns=names)


def _finite(value):
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'unavailable' if math.isnan(value) else ('infinity' if value > 0 else '-infinity')
    return value


def text_serialization(tables, dp, reference):
    data = tensor_serialization(tables, dp, reference)
    public = {}
    for name, array in data['arrays'].items():
        rows = array.reshape((-1, array.shape[-1])).tolist() if array.ndim > 1 else None
        if rows is not None:
            values, indices, lookup = [], [], {}
            for row in rows:
                signature = json.dumps(_finite(row), separators=(',', ':'))
                if signature not in lookup:
                    lookup[signature] = len(values); values.append(_compact_row(_finite(row)))
                indices.append(lookup[signature])
            public[name.replace('_', ' ')] = dict(shape=list(array.shape), rows=values, order=indices)
        else:
            public[name.replace('_', ' ')] = _finite(array.tolist())
    return ('Keep at least '+str(math.ceil(.85*len(tables.tasks)-1e-9))+' tasks on time. Energy is in joules; times are slots.\n'
            'S keeps a task on time; G gives no on-time promise. L is local; P/Q/R are stations. A-H are time bins; X is highest priority.\n'
            +json.dumps(dict(columns=data['columns'], values=public), sort_keys=True, separators=(',', ':'), allow_nan=False)
            +'\nReference:\n'+commitment_text(reference, tables)+'\n')


def _compact_row(row):
    if len(row) < 8 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or int(x) != x for x in row):
        return row
    runs = []; start = 0
    while start < len(row):
        step = row[start+1]-row[start] if start+1 < len(row) else 0
        end = start+1
        while end < len(row) and row[end] == row[start]+(end-start)*step:
            end += 1
        runs.append([end-start, row[start], step]); start = end
    return dict(runs=runs) if len(json.dumps(runs)) < len(json.dumps(row)) else row
