import argparse
import json
from pathlib import Path
import time
import noref_common as c


def generate(policy, rr, registry, *, emit=None):
    from lawn_mec.llm.grammar import CanonicalOptions, DESIGN_OPTIONS
    from lawn_mec.llm.sampler import TransformersSampler
    from lawn_mec.vcpm.library import pack
    from noref_input import spec, build_prompt
    grammar = spec(registry); mapping = CanonicalOptions.build(policy.tokenizer, DESIGN_OPTIONS)
    sampler = TransformersSampler(policy.model, policy.tokenizer, mapping, grammar)
    result = []
    for j,r in enumerate(rr):
        tic = time.perf_counter(); ctx = c.context(r); plan = None; error = None
        prompt = build_prompt(policy.tokenizer, r['key'], str(registry))
        try:
            draft = sampler.sample(r['key'], prompt, n=1, mode=True, seed=0)[0]
            plan = pack(grammar.build(r['key']).decode(draft.trace.prefix))
        except (ValueError, IndexError, KeyError, FloatingPointError) as e:
            error = str(e)
        check = c.validate(ctx, plan)
        row = dict(instance_key=r['key'], instance_sha256=r['sha256'], ordinal=j, plan=plan, check=check,
                   agreement=c.agreement(ctx,plan), decoder_error=error, wall_s=time.perf_counter()-tic)
        result.append(row)
        if emit:
            emit(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=c.SEEDS,required=True); p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--split',choices=('tune','validation','test'),required=True)
    p.add_argument('--adapter',required=True); p.add_argument('--out',help='default: <runs-root>/generation_tf/<split>/s<seed>')
    c.add_path_args(p,model=True); a=c.configure(p.parse_args()); a.out=a.out or str(c.TASK/'generation_tf'/a.split/f's{a.seed}')
    c.runtime(gpu=a.gpu)
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from lawn_mec.llm.policy import LoRAPolicy
    torch.set_num_threads(8); torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise ValueError('assigned GPU unavailable')
    adapter=Path(a.adapter); complete=json.loads((adapter.parent/'complete.json').read_text())
    if complete['seed']!=a.seed or not complete['complete'] or complete['adapter_sha256']!=c.sha(adapter/'adapter_model.safetensors'):
        raise ValueError('SFT completion/adapter guard differs')
    m=c.manifest(a.split); out=c.output(Path(a.out)/'samples.jsonl').parent
    cfg=dict(seed=a.seed,split=a.split,manifest_sha256=m['manifest_sha256'],
        adapter_sha256=complete['adapter_sha256'],epochs=complete['epochs'],code_version=c.VERSION,variant='no_reference',mode='field_argmax',
        scripts={n:c.sha(c.LIB/n) for n in ('noref_sample.py','noref_input.py','noref_common.py')})
    ch=c.pin(out/'config.json',cfg); reg=out/'registry.json'; c.pin(reg,{r['key']:r['path'] for r in m['instances']})
    with c.lease(out/'writer.lock'):
        old=c.rows(out/'samples.jsonl') if (out/'samples.jsonl').exists() else []
        done={r['instance_sha256']:r for r in old}
        if len(done)!=len(old) or any(r['config_sha256']!=ch for r in old):
            raise ValueError('duplicate/stale sampling row')
        expected={r['sha256']:(j,r['key']) for j,r in enumerate(m['instances'])}
        if any(h not in expected or (r['ordinal'],r['instance_key'])!=expected[h] for h,r in done.items()):
            raise ValueError('foreign sampling identity')
        policy=LoRAPolicy.load(c.MODEL,device='cuda',precision='float16')
        set_peft_model_state_dict(policy.model,load_file(adapter/'adapter_model.safetensors'),adapter_name='default')
        for j,r in enumerate(m['instances']):
            if r['sha256'] in done:
                continue
            row=generate(policy,[r],reg)[0]; row.update(ordinal=j,llm_seed=a.seed,split=a.split,config_sha256=ch)
            c.append(out/'samples.jsonl',row); done[r['sha256']]=row
            print('NOREF_SAMPLE',a.split,a.seed,j+1,row['check']['valid'],round(row['wall_s'],2),flush=True)
        stats=c.stats([done[r['sha256']] for r in m['instances']])
        c.write_json(out/'stats.json',dict(stats,seed=a.seed,split=a.split,
            config_sha256=ch,samples_sha256=c.sha(out/'samples.jsonl'),complete=len(done)==len(expected)))
        print('NOREF_SAMPLE_DONE',stats,flush=True)


if __name__=='__main__':
    main()
