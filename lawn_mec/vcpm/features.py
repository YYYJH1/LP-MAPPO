from dataclasses import dataclass
import numpy as np

from lawn_mec.marl_v2.env import token
from lawn_mec.marl_v2.state_schema import STATE_DIM, MAX_U, MAX_K, MAX_M, MAX_NODE
from lawn_mec.vcp.features import static_features

PUBLIC_NAMES = tuple(sorted((
    'free_energy_J', 'all_keep_energy_J', 'leave_one_out_energy_J', 'visits',
    'uploads', 'readiness', 'readiness_by_location', 'visit_bins', 'upload_bins',
    'minimum_travel', 'rush_marginal_J', 'upload_duration', 'line_of_sight',
    'mec_duration', 'local_duration', 'reference_layers', 'reference_tasks')))
COMMITMENT_DIM = MAX_U * 8 + MAX_K * 32
CRITIC_DIM = STATE_DIM + COMMITMENT_DIM


@dataclass(frozen=True)
class FactoredObservation:
    dynamic: np.ndarray
    public: np.ndarray

    @property
    def shape(self):
        return (len(self.dynamic)+len(self.public), 48)

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        return np.concatenate([self.dynamic, self.public]).astype(dtype or np.float32, copy=False)

    def __getitem__(self, index):
        return np.asarray(self)[index]


def signed_log(x):
    return float(np.sign(x) * np.log1p(abs(x)))


def public_tokens(arrays):
    if set(arrays) != set(PUBLIC_NAMES):
        raise ValueError('F(x) array inventory differs from vcpm schema')
    rows = []
    for group, name in enumerate(PUBLIC_NAMES):
        array = np.asarray(arrays[name], dtype=np.float64)
        if array.ndim > 4:
            raise ValueError('public array rank exceeds four')
        values = array.ravel(); i = 0
        shape = [*array.shape, *([0] * (4-array.ndim))]
        while i < len(values):
            x = values[i]
            status = 1 if np.isfinite(x) else (2 if np.isposinf(x) else 3 if np.isneginf(x) else 4)
            step = 0.
            if status == 1 and i+1 < len(values) and np.isfinite(values[i+1]):
                step = float(values[i+1] - x)
            end = i+1
            while end < len(values):
                y = values[end]
                equal = (y == x + (end-i)*step) if status == 1 else (
                    np.isnan(y) if status == 4 else y == x)
                if not equal:
                    break
                end += 1
            rows.append(token(6, [0, group, i, end-i, array.ndim, *shape,
                                  0 if status != 1 else signed_log(x), status,
                                  signed_log(step)]))
            i = end
    return np.stack(rows)


def build_public(context):
    return public_tokens(static_features(context.tables, context.dp, context.reference))


def interval(tables, k, field, choice):
    if choice is None:
        return (-1, -1)
    bins = tables.upload_bins[k] if field == 'upload' else tables.visit_bins[k, field]
    return tuple(map(int, bins[choice]))


def own_tokens(tables, commitment, u, slot):
    rows = [token(6, [1, u/MAX_U, commitment.layers[u]/7])]
    for k in tables.own[u]:
        c = commitment.tasks[k]
        spans = [interval(tables, k, f, getattr(c, f)) for f in ('s', 'a', 'upload')]
        times = [v/tables.N if v >= 0 else -1 for span in spans for v in span]
        slack = [(v-slot)/tables.N if v >= 0 else -1 for span in spans for v in span]
        rows.append(token(6, [2, k/MAX_K, float(c.g == 'S'),
                              -1 if c.location is None else c.location/(MAX_M+1),
                              *times, 'XYZ'.index(c.priority)/2, *slack]))
    return np.stack(rows)


def intent_tokens(tables, commitment, u):
    hist = np.zeros((tables.p['M'], 12), np.float64)
    rows = []
    for k, (task, c) in enumerate(zip(tables.tasks, commitment.tasks)):
        if c.g != 'S':
            continue
        if c.location and c.upload is not None:
            lo, hi = interval(tables, k, 'upload', c.upload)
            if hi >= lo:
                slots = np.arange(lo, hi+1)
                segments = np.minimum(11, slots*12//max(1, tables.N))
                hist[c.location-1] += np.bincount(segments, minlength=12)*task['D']/(hi-lo+1)/1e8
        if task['u'] != u:
            for field in ('s', 'a'):
                lo, hi = interval(tables, k, field, getattr(c, field))
                rows.append(token(6, [4, task['z_'+field]/MAX_NODE, task['u']/MAX_U,
                                      k/MAX_K, float(field == 'a'), lo/tables.N, hi/tables.N]))
    rows.extend(token(6, [3, m/MAX_M, *h]) for m, h in enumerate(hist))
    rows.extend(token(6, [5, v/MAX_U, sum(commitment.tasks[k].g == 'S' for k in own)/MAX_K])
                for v, own in enumerate(tables.own) if v != u)
    return np.stack(rows) if rows else np.zeros((0, 48), np.float32)


def encode_commitment(tables, commitment):
    if tables.p['U'] > MAX_U or len(tables.tasks) > MAX_K or tables.p['M'] > MAX_M:
        raise ValueError('commitment exceeds fixed F13 maxima')
    out = np.zeros(COMMITMENT_DIM, np.float32)
    if commitment is None:
        return out
    if len(commitment.layers) != tables.p['U'] or len(commitment.tasks) != len(tables.tasks):
        raise ValueError('commitment entity count mismatch')
    for u, layer in enumerate(commitment.layers):
        if not 0 <= layer < 7:
            raise ValueError('invalid layer')
        out[8*u] = 1; out[8*u+1+layer] = 1
    for k, (task, c) in enumerate(zip(tables.tasks, commitment.tasks)):
        row = np.zeros(32, np.float32)
        row[0] = 1; row[1+task['u']] = 1; row[7 + int(c.g == 'S')] = 1
        if c.location is not None:
            row[9+c.location] = 1
        for j, f in enumerate(('s', 'a', 'upload')):
            row[18+2*j:20+2*j] = [v/tables.N if v >= 0 else -1 for v in interval(tables, k, f, getattr(c, f))]
        row[24+'XYZ'.index(c.priority)] = 1
        out[MAX_U*8+k*32:MAX_U*8+(k+1)*32] = row
    return out
