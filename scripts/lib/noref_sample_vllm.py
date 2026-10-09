import argparse
import json
from pathlib import Path
import time
import noref_common as c


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=c.SEEDS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--split', choices=('tune','validation','test'), required=True)
    p.add_argument('--adapter', required=True); p.add_argument('--out', help='default: <runs-root>/generation/<split>/s<seed>')
    p.add_argument('--shards', type=int, default=1); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--merge-shards', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gpu-memory-utilization', type=float, default=.25)
    p.add_argument('--max-model-len', type=int, default=16384)
    c.add_path_args(p, model=True)
    return p


def selected_rows(rows, shards, index):
    if shards < 1 or not 0 <= index < shards:
        raise ValueError('require shards >= 1 and 0 <= shard-index < shards')
    return [(j,r) for j,r in enumerate(rows) if j % shards == index]


def adapter_guard(path, seed):
    adapter = Path(path)
    complete = json.loads((adapter.parent/'complete.json').read_text())
    actual = c.sha(adapter/'adapter_model.safetensors')
    if (complete['seed'] != seed or not complete['complete'] or complete['adapter_sha256'] != actual
        or complete['epochs'] not in (2,6)):
        raise ValueError('SFT completion/adapter guard differs')
    cfg = json.loads((adapter/'adapter_config.json').read_text())
    from lawn_mec.llm.policy import LORA_TARGETS
    if (cfg.get('r') != 64 or cfg.get('lora_alpha') != 128 or cfg.get('lora_dropout') != 0
        or set(cfg.get('target_modules', [])) != set(LORA_TARGETS) or cfg.get('bias') != 'none'):
        raise ValueError('registered LoRA 64/128 configuration differs')
    return complete


def serving_scripts():
    return {n:c.sha(c.LIB/n) for n in
            ('noref_sample_vllm.py','noref_vllm_core.py','noref_common.py','noref_input.py')}


def prepare(a):
    complete = adapter_guard(a.adapter, a.seed)
    c.runtime(gpu=None if a.merge_shards else a.gpu)
    m = c.manifest(a.split)
    if a.shards > len(m['instances']):
        raise ValueError('shards cannot exceed instance count')
    serving = dict(backend='vllm-field', dtype='float16', batch_size=a.batch_size, shards=a.shards,
        gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
        tie_rule='first maximum in grammar option order',
        scripts=serving_scripts())
    out = Path(a.out)
    cfg = dict(seed=a.seed,split=a.split,manifest_sha256=m['manifest_sha256'],
        adapter_sha256=complete['adapter_sha256'],epochs=complete['epochs'],code_version=c.VERSION,
        variant='no_reference',mode='field_argmax',
        scripts={n:c.sha(c.LIB/n) for n in ('noref_sample.py','noref_input.py','noref_common.py')},
        serving=serving)
    out = c.output(out/'config.json').parent
    reg = out/'registry.json'
    import fcntl
    with c.output(out/'metadata.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ch = c.pin(out/'config.json',cfg)
        c.pin(reg,{r['key']:r['path'] for r in m['instances']})
    return complete,m,out,cfg,ch,reg


def check_rows(records, selected, ch, a):
    done = {r['instance_sha256']:r for r in records}
    expected = {r['sha256']:(j,r['key']) for j,r in selected}
    if len(done) != len(records):
        raise ValueError('duplicate sampling row')
    for h,r in done.items():
        if (h not in expected or (r['ordinal'],r['instance_key']) != expected[h]
            or r['config_sha256'] != ch or r['llm_seed'] != a.seed
            or r.get('split') != a.split):
            raise ValueError('stale/foreign sampling row')
    return done


def verify_adapter(a, complete):
    if adapter_guard(a.adapter,a.seed) != complete:
        raise ValueError('STOP: adapter/completion changed during generation')


def finish_output(root, records, diagnostics, a, m, cfg, ch, *, complete=True):
    for name, rows in (('samples.jsonl',records),('fields.jsonl',diagnostics)):
        part = c.output(root/(name+'.part'))
        with part.open('w') as f:
            for row in rows:
                f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            f.flush(); __import__('os').fsync(f.fileno())
        part.replace(c.output(root/name))
    stats = c.stats(records)
    c.write_json(root/'stats.json',dict(stats,seed=a.seed,split=a.split,
        config_sha256=ch,samples_sha256=c.sha(root/'samples.jsonl'),complete=complete))
    print('NOREF_SAMPLE_DONE',stats,flush=True)


def merge(a, complete, m, out, cfg, ch, reg):
    rows, traces = [], []
    with c.lease(out/'writer.lock'):
        for index in range(a.shards):
            root = out/'shards'/f'sh{index}'
            marker = json.loads((root/'shard.json').read_text())
            stats = json.loads((root/'stats.json').read_text())
            if marker != dict(shards=a.shards,shard_index=index,config_sha256=ch):
                raise ValueError('mixed shard configuration')
            if (not stats['complete'] or stats['config_sha256'] != ch
                or stats['samples_sha256'] != c.sha(root/'samples.jsonl')):
                raise ValueError('incomplete/changed shard')
            records = c.rows(root/'samples.jsonl')
            selected = selected_rows(m['instances'],a.shards,index)
            if len(check_rows(records,selected,ch,a)) != len(selected):
                raise ValueError('shard does not have exact coverage')
            rows.extend(records); traces.extend(c.rows(root/'fields.jsonl'))
        if len(check_rows(rows,list(enumerate(m['instances'])),ch,a)) != len(m['instances']):
            raise ValueError('merged coverage differs')
        rows.sort(key=lambda r:r['ordinal']); traces.sort(key=lambda r:r['ordinal'])
        validate_traces(rows,traces,ch)
        verify_adapter(a,complete)
        from copy import copy
        full = copy(a); full.shards = 1
        finish_output(out,rows,traces,full,m,cfg,ch)
    print('NOREF_MERGE_PASS',len(rows),a.shards,flush=True)


def validate_traces(rows,traces,ch):
    expected = {(r['ordinal'],r['instance_key'],r['instance_sha256']) for r in rows}
    actual = {(r['ordinal'],r['instance_key'],r['instance_sha256']) for r in traces}
    if len(traces) != len(actual) or actual != expected or any(r['config_sha256'] != ch for r in traces):
        raise ValueError('field diagnostics coverage/config differs')


def run(a):
    complete,m,out,cfg,ch,reg = prepare(a)
    if a.merge_shards:
        merge(a,complete,m,out,cfg,ch,reg); return
    root = out if a.shards == 1 else c.output(out/'shards'/f'sh{a.shard_index}'/'samples.jsonl').parent
    if a.shards != 1:
        c.pin(root/'shard.json',dict(shards=a.shards,shard_index=a.shard_index,config_sha256=ch))
    selected = selected_rows(m['instances'],a.shards,a.shard_index)
    with c.lease(root/'writer.lock'):
        old = c.rows(root/'samples.jsonl') if (root/'samples.jsonl').exists() else []
        done = check_rows(old,selected,ch,a)
        traces = c.rows(root/'fields.jsonl') if (root/'fields.jsonl').exists() else []
        validate_traces(old,traces,ch)
        trace_index = {r['instance_sha256']:r for r in traces}
        todo = [(j,r) for j,r in selected if r['sha256'] not in done]
        if todo:
            from transformers import AutoTokenizer
            from noref_input import build_prompt, spec
            from lawn_mec.llm.prompt import token_budget
            from lawn_mec.vcpm.library import pack
            from noref_vllm_core import create
            tokenizer = AutoTokenizer.from_pretrained(c.MODEL,local_files_only=True)
            sampler, decoder = create(tokenizer,reg,a.adapter,memory=a.gpu_memory_utilization,
                                      max_model_len=a.max_model_len)
            with sampler:
                for offset in range(0,len(todo),a.batch_size):
                    verify_adapter(a,complete)
                    batch = todo[offset:offset+a.batch_size]; tic = time.perf_counter()
                    requests = [(r['key'],build_prompt(tokenizer,r['key'],str(reg))) for _,r in batch]
                    for key,prompt in requests:
                        budget = token_budget(tokenizer,prompt,spec(reg).build(key),a.max_model_len)
                        if not budget['fits']:
                            raise RuntimeError('full context exceeds --max-model-len: '+key+' '+str(budget))
                    states = decoder.generate(requests)
                    if len(states) != len(batch):
                        raise RuntimeError('incomplete instance batch')
                    for (j,r),(_,prompt),state in zip(batch,requests,states):
                        ctx = c.context(r); plan = None; error = state.error
                        if error is None:
                            try:
                                plan = pack(state.grammar.decode(state.prefix))
                            except (ValueError,IndexError,KeyError,FloatingPointError) as exc:
                                error = str(exc)
                        row = dict(instance_key=r['key'],instance_sha256=r['sha256'],ordinal=j,plan=plan,
                            check=c.validate(ctx,plan),agreement=c.agreement(ctx,plan),decoder_error=error,
                            wall_s=time.perf_counter()-tic,llm_seed=a.seed,config_sha256=ch)
                        row.update(split=a.split)
                        trace = dict(instance_key=r['key'],instance_sha256=r['sha256'],ordinal=j,
                            config_sha256=ch,fields=state.fields)
                        c.append(root/'fields.jsonl',trace); c.append(root/'samples.jsonl',row)
                        done[r['sha256']] = row; trace_index[r['sha256']] = trace
                        print('NOREF_SAMPLE',a.split,a.seed,j+1,row['check']['valid'],
                              round(row['wall_s'],2),flush=True)
        records = [done[r['sha256']] for _,r in selected]
        traces = [trace_index[r['sha256']] for _,r in selected]
        verify_adapter(a,complete)
        finish_output(root,records,traces,a,m,cfg,ch)


def main():
    p = parser(); a = c.configure(p.parse_args())
    a.out = a.out or str(c.TASK/'generation'/a.split/f's{a.seed}')
    if (a.gpu < 0 or a.batch_size < 1 or a.max_model_len < 2
        or not 0 < a.gpu_memory_utilization <= 1 or a.shards < 1 or not 0 <= a.shard_index < a.shards):
        p.error('invalid GPU, batch, context, memory utilization or shard arguments')
    run(a)


if __name__ == '__main__':
    main()
