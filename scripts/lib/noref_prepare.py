import argparse
from pathlib import Path
import time
import noref_common as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', help='default: <runs-root>/planner/data'); p.add_argument('--allow-missing', action='store_true')
    c.add_path_args(p, model=True); a = c.configure(p.parse_args()); a.out = a.out or str(c.TASK/'planner/data'); c.runtime()
    from transformers import AutoTokenizer
    from lawn_mec.vcpm.library import pack
    from lawn_mec.vcp.serialize import commitment_text
    from noref_input import payload
    from lawn_mec.llm.prompt_compact import serialize_prompt
    from lawn_mec.llm.grammar import CanonicalOptions, DESIGN_OPTIONS, encode_trace
    from noref_input import spec
    tok = AutoTokenizer.from_pretrained(c.MODEL, local_files_only=True, trust_remote_code=False)
    mapping = CanonicalOptions.build(tok, DESIGN_OPTIONS)
    m = c.manifest('train'); rr = m['instances']; out = c.output(Path(a.out)/'train.jsonl').parent
    cfg = dict(split='train', variant='no_reference', single_label='nominal DP full plan',
               manifest_sha256=m['manifest_sha256'], code_version=c.VERSION,
               input_sources={name:c.sha(c.LIB/name) for name in ('noref_input.py','noref_common.py')})
    ch = c.pin(out/'config.json', cfg)
    reg = out/'registry.json'; c.pin(reg, {r['key']:r['path'] for r in rr})
    existing = c.rows(out/'train.jsonl') if (out/'train.jsonl').exists() else []
    done = {r['instance_sha256']:r for r in existing}
    allowed = {r['sha256'] for r in rr}
    if len(done) != len(existing) or not set(done) <= allowed or any(r['config_sha256'] != ch for r in existing):
        raise ValueError('duplicate/foreign/stale prepared rows')
    missing = []; lengths = []; tic = time.perf_counter()
    for j,r in enumerate(rr):
        if r['sha256'] in done:
            lengths.append(done[r['sha256']]['sequence_length']); continue
        if c.cached_path(r) is None:
            missing.append(r['key']); continue
        ctx = c.context(r); plan = pack(ctx.reference); check = c.validate(ctx, plan)
        if not check['valid']:
            raise ValueError('DP teacher invalid: '+r['key'])
        public = payload(ctx); prompt = str(serialize_prompt(tok, public, variant='no_reference'))
        target = commitment_text(ctx.reference, ctx.tables)
        if 'DP-Commit reference' in prompt or target in prompt or any(k.startswith('reference') for k in public):
            raise AssertionError('DP leakage')
        tokens = list(ctx.reference.tokens(ctx.tables))
        trace = encode_trace(tok, mapping, spec(reg), r['key'], prompt, tuple(tokens))
        if len(trace.token_ids) > 8192:
            raise ValueError('overlength; never truncate training')
        row = dict(config_sha256=ch, instance_key=r['key'], instance_path=r['path'], instance_sha256=r['sha256'],
                   ordinal=j, prompt=prompt, target=target, plan=plan, tokens=tokens,
                   sequence_length=len(trace.token_ids), variant='no_reference', single_label=True)
        c.append(out/'train.jsonl', row); done[r['sha256']]=row; lengths.append(len(trace.token_ids))
        if len(done)%100 == 0:
            print('NOREF_PREPARE_PROGRESS', len(done), 'wall_s', round(time.perf_counter()-tic,1), flush=True)
    result = dict(instances=len(done), expected=len(rr), missing=missing, complete=not missing,
        train_sha256=c.sha(out/'train.jsonl') if done else None, config_sha256=ch, wall_s=time.perf_counter()-tic,
        max_sequence_length=max(lengths,default=0), train_only=True, no_holdout=True, single_label=True)
    c.write_json(out/'complete.json', result); print('NOREF_PREPARE',result,flush=True)
    if missing and not a.allow_missing:
        raise SystemExit('Missing contexts; build them, then rerun to append the remaining train instances.')


if __name__ == '__main__':
    main()
