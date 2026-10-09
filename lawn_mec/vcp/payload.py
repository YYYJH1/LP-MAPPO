import hashlib
import json
import math
import random
import numpy as np
from lawn_mec.llm.vcp_grammar import get_context, resolve_reference_rate
from lawn_mec.vcp.reference_rate import DEFAULT_REFERENCE_RATE

ANONYMOUS_SEED = 20260929
VARIANTS = ("full", "anonymous", "no_reference")
SCHEMA = (
    ("peak", "idlePeak", "idleTrough"),
    ("battery", "free", "rush", "LoS"),
    ("s", "a", "deadline", "freshness", "data", "work", "local",
     "upload", "LoS", "MEC", "margin", "ul", "layer"),
)


def number(value):
    value = float(value)
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, ".3g")


def boundaries(intervals):
    intervals = [(int(lo), int(hi)) for lo, hi in intervals]
    if any(intervals[j][1]+1 != intervals[j+1][0] for j in range(7)):
        raise ValueError("V1 bins must be contiguous")
    return " ".join(str(lo) for lo, _ in intervals)+" "+str(intervals[-1][1]+1)


def _runs(values, labels):
    rows = []
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[end] == values[start]:
            end += 1
        label = labels[start] if end == start+1 else labels[start]+"-"+labels[end-1]
        rows.append(label+":"+values[start])
        start = end
    return ";".join(rows)


def _intervals(indices):
    indices = list(map(int, indices))
    parts = []
    while indices:
        start = end = indices.pop(0)
        while indices and indices[0] == end+1:
            end = indices.pop(0)
        parts.append(f"{start}:{end+1}")
    return ",".join(parts)


def _build_payload(instance_key, registry):
    from lawn_mec.vcp.features import static_features
    from lawn_mec.vcp.grammar import Grammar
    from lawn_mec.vcp.serialize import commitment_text

    context = get_context(instance_key, registry)
    tb = context.tables
    features = static_features(tb, context.dp, context.reference)
    bs = []
    for m, caps in enumerate(tb.cpu_capacity):
        bs.append(("PQR"[m], (
            _intervals(np.flatnonzero(caps == np.min(caps))),
            number(np.min(caps)/tb.p["delta"]/1e9),
            number(np.max(caps)/tb.p["delta"]/1e9))))
    uavs, tasks = [], []
    for u, own in enumerate(tb.own):
        los = np.mean(features["line_of_sight"][list(own)], axis=0).T
        uavs.append((f"U{u+1}", (
            number(tb.p["battery"]/1000), number(features["free_energy_J"][u]/1000),
            number(features["all_keep_energy_J"][u]/1000),
            _runs(["/".join(number(x) for x in row) for row in los], "ABCDEFG"))))
        for k in own:
            t = tb.tasks[k]
            layers = np.argmax(features["line_of_sight"][k], axis=1)
            upload = [features["upload_duration"][k, m, ell] for m, ell in enumerate(layers)]
            sight = [features["line_of_sight"][k, m, ell] for m, ell in enumerate(layers)]
            tasks.append((f"T{k+1}", (
                boundaries(features["visit_bins"][k, 0]), boundaries(features["visit_bins"][k, 1]),
                str(math.floor(t["d"]/tb.p["delta"])), str(math.floor(t["H"]/tb.p["delta"])),
                number(t["D"]/1e6), number(t["C"]/1e9), str(int(features["local_duration"][k])),
                "/".join(str(int(x)) for x in upload), "".join("L" if x else "N" for x in sight),
                "/".join(str(int(x)) for x in features["mec_duration"][k]),
                number(features["rush_marginal_J"][k]/1000), boundaries(features["upload_bins"][k]),
                "".join("ABCDEFG"[ell] for ell in layers))))
    return dict(tables=(bs, uavs, tasks), schema=SCHEMA, features=features,
                instance_sha256=context.sha256, reference=commitment_text(context.reference, tb),
                c_min=Grammar(tb).c_min, K=len(tb.tasks), g_cap=len(tb.tasks)-Grammar(tb).c_min, context=context,
                owners=tuple(tuple(f"T{k+1}" for k in own) for own in tb.own))


VERSION = 'planner-compact-v1'
SCHEMA_SPEC = dict(version=VERSION, columns=SCHEMA,
    units=['slots', 'kJ', 'Mbit', 'Gcycle', 'GHz'], significant_digits=3,
    time='floor slots; nine exact half-open bin boundaries',
    los='owned acquisition-node mean; first best-LoS task layer',
    static=['tables', 'schema', 'owners', 'reference', 'c_min', 'K', 'g_cap'],
    legal='shared exact prefix oracle; not in static learned inputs',
    encoding='decimal coefficient/exponent and UTF-8 literal bytes v1')
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA_SPEC, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()


REFERENCE_RATE_FIELD = 'payload_reference_rate'


def provenance(reference_rate=None):
    record = dict(payload_schema_version=VERSION, payload_schema_sha256=SCHEMA_SHA256)
    rate = resolve_reference_rate(reference_rate)
    if rate != DEFAULT_REFERENCE_RATE:
        record[REFERENCE_RATE_FIELD] = rate
    return record


def require_provenance(record, reference_rate=None):
    expected = provenance(reference_rate)
    if any(record.get(k) != v for k, v in expected.items() if k != REFERENCE_RATE_FIELD):
        raise ValueError('payload schema changed or missing; regenerate SFT/GRPO provenance')
    found = record.get(REFERENCE_RATE_FIELD, DEFAULT_REFERENCE_RATE)
    wanted = expected.get(REFERENCE_RATE_FIELD, DEFAULT_REFERENCE_RATE)
    if found != wanted:
        raise ValueError(f'payload reference rate {found} differs from {wanted}; regenerate SFT/GRPO provenance')


def prompt_payload(instance_key, registry):
    data = _build_payload(instance_key, registry)
    tb = data['context'].tables
    data['legal'] = dict(
        M=int(tb.p['M']), U=int(tb.p['U']), N=int(tb.N),
        own=[list(own) for own in tb.own], visits=tb.instance['visits'],
        tasks=[dict(u=int(t['u']), L_s=int(t['L_s']), L_a=int(t['L_a']),
                    z_s=int(t['z_s']), z_a=int(t['z_a']),
                    d=math.floor(t['d']/tb.p['delta']),
                    H=math.floor(t['H']/tb.p['delta'])) for t in tb.tasks],
        visit_bins=[[[list(pair) for pair in tb.visit_bins[k, phase]]
                     for phase in ('s', 'a')] for k in range(len(tb.tasks))],
        upload_bins=[[list(pair) for pair in tb.upload_bins[k]] for k in range(len(tb.tasks))],
        ready_min=tb.ready_min.tolist(), min_time=tb.min_time.tolist())
    data.update(provenance(tb.reference_rate))
    return data


def learned_payload(data):
    return {key: data[key] for key in SCHEMA_SPEC['static']}


def grammar_tables(data):
    from types import SimpleNamespace
    legal = data['legal']
    return SimpleNamespace(p=dict(M=legal['M'], U=legal['U'], delta=1),
        N=legal['N'], own=legal['own'], tasks=legal['tasks'],
        instance=dict(visits=legal['visits']),
        visit_bins={(k, phase): tuple(map(tuple, bins))
                    for k, pair in enumerate(legal['visit_bins'])
                    for phase, bins in zip(('s', 'a'), pair)},
        upload_bins={k: tuple(map(tuple, bins)) for k, bins in enumerate(legal['upload_bins'])},
        ready_min=np.asarray(legal['ready_min']), min_time=np.asarray(legal['min_time']))


CONTROLS = {
    'NCP': dict(input='compact', primary=True, masks='shared-exact'),
    'NCP-full': dict(input='full-arrays', primary=False, venue='secondary',
                     difference='unrounded full arrays instead of compact cells'),
    'shared-exact-mask': dict(primary=False, venue='secondary',
        difference='legal source tables absent from static text and compact tensors; '
                   'both decoders receive identical prefix-conditioned masks'),
}
