import argparse
from pathlib import Path
import json
import noref_common as c


def convert(r, raw):
    ctx=c.context(r); check=c.validate(ctx,raw['plan'])
    if check!=raw['check']:
        raise ValueError('sampling grammar/precheck proof changed')
    plan,mode=c.execution(raw['plan'],check)
    return dict(instance_key=r['key'],instance_sha256=r['sha256'],ordinal=raw['ordinal'],
        plan=raw['plan'],execution_plan=plan,interface=mode,check=check,
        agreement=c.agreement(ctx,raw['plan']),invalid_fallback='L/f17' if not check['valid'] else None)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,choices=c.SEEDS,required=True)
    p.add_argument('--split',choices=('tune','validation','test'),required=True)
    p.add_argument('--source',required=True); p.add_argument('--out',help='default: <runs-root>/plans/<split>/s<seed>')
    c.add_path_args(p); a=c.configure(p.parse_args()); a.out=a.out or str(c.TASK/'plans'/a.split/f's{a.seed}')
    c.runtime(); m=c.manifest(a.split); raw=c.rows(a.source)
    cfg=json.loads(Path(a.source).with_name('config.json').read_text()); stats=json.loads(Path(a.source).with_name('stats.json').read_text())
    if cfg['seed']!=a.seed or cfg['split']!=a.split or cfg['manifest_sha256']!=m['manifest_sha256'] or not stats['complete'] or stats['samples_sha256']!=c.sha(a.source):
        raise ValueError('sampling completion/source identity differs')
    indexed={r['instance_sha256']:r for r in raw}
    if len(indexed)!=len(raw) or set(indexed)!={r['sha256'] for r in m['instances']}:
        raise ValueError('exact split coverage required')
    out=c.output(Path(a.out)/'plans.jsonl').parent
    configuration=dict(seed=a.seed,split=a.split,source_sha256=c.sha(a.source),
        manifest_sha256=m['manifest_sha256'],code_version=c.VERSION,adapter_sha256=cfg['adapter_sha256'],epochs=cfg['epochs'],
        conversion='full plan; no transplant, anchor, repair, or DP fallback',
        scripts={n:c.sha(c.LIB/n) for n in ('noref_convert.py','noref_common.py')})
    ch=c.pin(out/'config.json',configuration)
    with c.lease(out/'writer.lock'):
        old=c.rows(out/'plans.jsonl') if (out/'plans.jsonl').exists() else []
        done={r['instance_sha256']:r for r in old}
        if len(done)!=len(old) or any(r['config_sha256']!=ch for r in old):
            raise ValueError('conversion resume differs')
        for j,r in enumerate(m['instances']):
            source=indexed[r['sha256']]
            if source['ordinal']!=j or source['instance_key']!=r['key'] or source['config_sha256']!=c.digest(cfg):
                raise ValueError('sampling row identity/config differs')
            row=dict(convert(r,source),schema='noref-oursm-v1',llm_seed=a.seed,split=a.split,config_sha256=ch)
            if r['sha256'] in done:
                if done[r['sha256']]!=row: raise ValueError('conversion row differs')
            else:
                c.append(out/'plans.jsonl',row); done[r['sha256']]=row
        c.write_json(out/'complete.json',dict(complete=True,plans_sha256=c.sha(out/'plans.jsonl'),
            config_sha256=ch,**c.stats(list(done.values()))))
        print('NOREF_CONVERT_DONE',a.split,a.seed,len(done),sum(not r['check']['valid'] for r in done.values()),flush=True)


if __name__=='__main__':main()
