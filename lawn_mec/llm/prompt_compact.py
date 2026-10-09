import random


from .prompt import token_budget
from lawn_mec.vcp.payload import (prompt_payload, number, boundaries,
                                   SCHEMA, ANONYMOUS_SEED, VARIANTS)


class CompactPrompt(str):
    def __new__(cls, value, metadata=None):
        obj = super().__new__(cls, value)
        obj.metadata = {} if metadata is None else metadata
        return obj


def _columns(variant, schema=SCHEMA):
    indices = list(range(20))
    random.Random(ANONYMOUS_SEED).shuffle(indices)
    columns, offset = [], 0
    for names in schema:
        row = [(name, j) for j, name in enumerate(names)]
        if variant == "anonymous":
            row = sorted(((f"f{indices[offset+j]+1}", j) for j in range(len(names))),
                         key=lambda pair: int(pair[0][1:]))
        columns.append(row)
        offset += len(names)
    return columns


def build_prompt(tokenizer, instance_key, registry, *, variant="full"):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    data = prompt_payload(instance_key, registry)
    return serialize_prompt(tokenizer, data, variant=variant)


def serialize_prompt(tokenizer, data, *, variant="full"):
    from lawn_mec.vcp.payload import provenance
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    columns = _columns(variant, data['schema'])
    system = (f"Minimize eta=total energy/on-time tasks. Require {data['c_min']}/{data['K']} "
              f"on time; battery never negative. At most {data['g_cap']} G unless forced by grammar. "
              "S promises on-time service; G does not. Output only UAV blocks in order: "
              "U<number> layer <A-G>, then T<number> g location s a upload priority in visit order. "
              "Distinct layers; location L=local,P/Q/R=BS; bins A-H; priority X>Y>Z. "
              "Local upload is '-'; G requires G - - - - Z.")
    lines = ["Units: slots,kJ,Mbit,Gcycle,GHz. Rounded display only; use exact grammar masks.",
             "s=acquisition,a=follow-up,ul=upload: nine boundaries; adjacent pairs are A-H half-open bins (right excluded). "
             "P/Q/R vector order. L/N=LoS/NLoS; upload uses listed best-LoS layers. "
             "MEC from earliest acquisition end; no contention. LoS fractions over acquisition nodes. "
             "free/rush=relaxed no/all-keep energy; margin=rush minus leave-one-out. Peak=max-load plateau."]
    from collections import Counter
    counts = Counter(row[j] for _, row in data["tables"][2] for j in (0, 1, 11))
    repeated = {value: "@"+chr(97+i) for i, value in enumerate(v for v, count in counts.items() if count > 1)}
    if repeated:
        lines.append("Shared bin boundaries:")
        lines.extend(label+"="+value for value, label in repeated.items())
    for title, rows, cols in zip(("BS", "UAV", "Tasks (visit order)"), data["tables"], columns):
        lines.append(title+"|"+"|".join(name for name, _ in cols))
        for label, values in rows:
            if title.startswith("Tasks"):
                for u, own in enumerate(data["owners"]):
                    if label == own[0]:
                        lines.append(f"U{u+1}")
            lines.append(label+"|"+"|".join(repeated.get(values[j], values[j]) if title.startswith("Tasks") and j in (0, 1, 11)
                                           else values[j] for _, j in cols))
    if variant != "no_reference":
        lines.extend(("DP-Commit reference:", data["reference"]))
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return CompactPrompt(rendered, dict(variant=variant, anonymous_seed=ANONYMOUS_SEED,
        columns=columns, significant_digits=3, time_rounding="floor slots; exact half-open bins",
        instance_sha256=data["instance_sha256"], **provenance()))


def parse_prompt(text):
    import re
    requirement = re.search(r'Require (\d+)/(\d+) on time', text)
    if requirement is None:
        raise ValueError('missing public completion requirement')
    cap = re.search(r'At most (\d+) G unless forced', text)
    if cap is None:
        raise ValueError('missing public G cap')
    shared, tables, schema, owners, reference = {}, [], [], [], []
    in_reference = False
    for line in text.splitlines():
        line = line.split('<|im_end|>', 1)[0]
        if line.startswith('@') and '=' in line:
            name, value = line.split('=', 1); shared[name] = value
        elif line.startswith(('BS|', 'UAV|', 'Tasks (visit order)|')):
            tables.append([])
            schema.append(tuple(line.split('|')[1:]))
        elif line == 'DP-Commit reference:':
            in_reference = True
        elif in_reference:
            if re.fullmatch(r'U\d+ layer [A-G]|T\d+ [SG] [LPQR-] [A-H-] [A-H-] [A-H-] [XYZ]', line):
                reference.append(line)
        elif len(tables) == 3 and re.fullmatch(r'U\d+', line):
            owners.append([])
        elif tables and '|' in line:
            label, *values = line.split('|')
            tables[-1].append((label, tuple(shared.get(v, v) for v in values)))
            if len(tables) == 3:
                owners[-1].append(label)
    if len(tables) != 3:
        raise ValueError('incomplete compact tables')
    return dict(tables=tuple(tables), schema=tuple(schema), owners=tuple(map(tuple, owners)),
                reference='\n'.join(reference), c_min=int(requirement[1]), K=int(requirement[2]), g_cap=int(cap[1]))
