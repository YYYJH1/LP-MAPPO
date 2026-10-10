"""Evaluate LP-MAPPO: the generated plans are executed by the trained MAPPO policies under the execution constraints.

  python scripts/evaluate.py --seed 42 --plans runs/plans/test/s42/plans.jsonl

Each instance is evaluated with the five MAPPO policies and two action seeds. Records are written to
runs/eval/<split>/K1/s<seed>.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent/'lib'))
import noref_common as c


def candidates(ctx,raw,ordinal,K):
    plan,mode=c.execution(raw['plan'],raw['check'])
    if not raw['check']['valid']:
        return [dict(id='invalid-L',plan=None,interface='L')],dict(invalid=True,attempted_edits=0)
    if K==1:
        return [dict(id='R',plan=plan,interface='R')],dict(invalid=False,attempted_edits=0)
    import numpy as np
    from lawn_mec.vcpm.library import pack,unpack
    from lawn_mec.vcpm.residual_labels import enumerate_edits
    from lawn_mec.vcp.grammar import Grammar
    edited_context=SimpleNamespace(tables=ctx.tables,dp=ctx.dp,reference=unpack(plan),grammar=Grammar(ctx.tables))
    raw_edits=[pack(p) for _,p in enumerate_edits(edited_context,budget=32,max_tasks=2,seed=203+ordinal)]
    order=np.random.default_rng(20261002+ordinal).permutation(len(raw_edits)).tolist()
    if len(raw_edits)<7:
        raise ValueError('J=10 requires seven admissible plan edits')
    pool=[dict(id=m,plan=plan if m!='L' else None,interface=m) for m in ('R','B','L')]
    valid=[]
    for i,idx in enumerate(order[:7]):
        p=raw_edits[idx]; check=c.validate(ctx,p); valid.append(check['valid'])
        if check['valid']:
            pool.append(dict(id='edit'+str(i+1),plan=p,interface='R'))
    return pool,dict(invalid=False,attempted_edits=7,available_edit_pool=len(raw_edits),
                     edit_order=order[:7],valid_edits=valid,generator_seed=203+ordinal,permutation_seed=20261002+ordinal,
                     tie_order=['R','B','L',*['edit'+str(i+1) for i in range(7)]])


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed',type=int,choices=c.SEEDS,required=True,help='LoRA seed of the planner whose plans are evaluated')
    p.add_argument('--split',choices=('tune','validation','test'),default='test')
    p.add_argument('--J',type=int,choices=(0,10),default=0,help='screening simulations of the optional plan verification (default 0: none)')
    p.add_argument('--plans',help='plans.jsonl (default: <data-root>/plans/<split>/s<seed>/plans.jsonl)')
    p.add_argument('--out',help='output directory (default: <runs-root>/eval/<split>/K<1|10>/s<seed>/sh<shard-index>)')
    p.add_argument('--workers',type=int,default=6); p.add_argument('--shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--mappo-seeds',type=int,nargs='+',choices=c.MAPPOS,help='subset of MAPPO policy seeds (default: all five)')
    p.add_argument('--instances',nargs='+',help='subset of instance keys, e.g. test_1400000 (default: all)')
    c.add_path_args(p); a=c.configure(p.parse_args())
    a.K=1 if a.J==0 else a.J
    a.plans=a.plans or str(c.DATA/'plans'/a.split/f's{a.seed}'/'plans.jsonl')
    a.out=a.out or str(c.TASK/'eval'/a.split/f'K{a.K}'/f's{a.seed}'/f'sh{a.shard_index:02d}')
    mappos=tuple(sorted(set(a.mappo_seeds))) if a.mappo_seeds else c.MAPPOS
    if not 1<=a.workers<=32 or not 0<=a.shard_index<a.shards or (a.K==10 and a.split!='test'):
        p.error('workers must be 1..32, the shard index below --shards, and J=10 is defined for the test split only')
    c.runtime(); m=c.manifest(a.split)
    import marl_plan_runtime as R
    rows=c.rows(a.plans); indexed={r['instance_sha256']:r for r in rows}; conf=json.loads(Path(a.plans).with_name('config.json').read_text())
    complete=json.loads(Path(a.plans).with_name('complete.json').read_text())
    if (len(indexed)!=len(rows) or set(indexed)!={r['sha256'] for r in m['instances']} or not complete['complete']
        or complete['plans_sha256']!=c.sha(a.plans) or conf['seed']!=a.seed or conf['split']!=a.split
        or conf['manifest_sha256']!=m['manifest_sha256']):
        raise ValueError('plans exact coverage/config/completion required')
    payload,proofs=R.load_runs(mappos)
    out=c.output(Path(a.out)/'results.jsonl').parent
    cfg=dict(seed=a.seed,split=a.split,K=a.K,manifest_sha256=m['manifest_sha256'],
        plans_sha256=c.sha(a.plans),adapter_sha256=conf['adapter_sha256'],epochs=conf['epochs'],checkpoints=proofs,code_version=c.VERSION,baseline=None,
        shards=a.shards,shard_index=a.shard_index,noise_bases=list(c.NOISE[a.split]),
        script_sha256=c.sha(__file__),runtime_sha256=c.sha(c.LIB/'marl_plan_runtime.py'))
    keys={r['key'] for r in m['instances']}
    if a.instances and not set(a.instances)<=keys:
        p.error('unknown instance keys: '+repr(sorted(set(a.instances)-keys)))
    if a.instances or a.mappo_seeds:
        cfg['subset']=dict(instances=sorted(a.instances) if a.instances else None,mappo_seeds=list(mappos))
    ch=c.pin(out/'config.json',cfg)
    selected=[(j,r) for j,r in enumerate(m['instances']) if j%a.shards==a.shard_index and (not a.instances or r['key'] in a.instances)]
    with c.lease(out/'writer.lock'):
        done=c.rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
        index={(r['instance_sha256'],r['mappo_seed'],r['action_seed']):r for r in done}
        expected={(r['sha256'],s,b+j) for j,r in selected for s in mappos for b in c.NOISE[a.split]}
        if len(index)!=len(done) or not set(index)<=expected or any(r['config_sha256']!=ch for r in done):
            raise ValueError('eval resume grid/config differs')
        choices=c.rows(out/'choices.jsonl') if (out/'choices.jsonl').exists() else []
        choice_index={(r['instance_sha256'],r['mappo_seed']):r for r in choices}
        if len(choice_index)!=len(choices) or any(r['config_sha256']!=ch for r in choices):
            raise ValueError('screen choice journal differs')
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=R.init,initargs=(payload,)) as pool:
            for j,r in selected:
                raw=indexed[r['sha256']]; ctx=c.context(r)
                if (raw['llm_seed']!=a.seed or raw['split']!=a.split or raw['ordinal']!=j or raw['schema']!='noref-oursm-v1'
                    or raw['config_sha256']!=c.digest(conf) or raw['check']!=c.validate(ctx,raw['plan'])):
                    raise ValueError('plan row/proof differs')
                candidate,meta=candidates(ctx,raw,j,a.K)
                pending=[]
                for s in mappos:
                    address=(r['sha256'],s)
                    if address not in choice_index:
                        if a.K==10:
                            tasks=[dict(row=r,plan=x['plan'],interface=x['interface'],mappo_seed=s,action_seed=c.SCREEN['test']+j) for x in candidate]
                            scores=list(pool.map(R.worker,tasks)); losses=[x['loss'] for x in scores]
                            if not all(math.isfinite(v) for v in losses):raise ValueError('nonfinite screening loss')
                            win=min(range(len(candidate)),key=lambda i:(losses[i],i))
                        else:
                            losses=None; win=0
                        choice=dict(config_sha256=ch,instance_sha256=r['sha256'],mappo_seed=s,
                            candidate=candidate[win],candidate_sha256=c.digest(candidate),meta=meta,screen_losses=losses,
                            actual_K=len(candidate),screening_seed=c.SCREEN['test']+j if a.K==10 else None)
                        c.append(out/'choices.jsonl',choice); choice_index[address]=choice
                    choice=choice_index[address]
                    if choice['candidate_sha256']!=c.digest(candidate) or choice['candidate'] not in candidate:
                        raise ValueError('resumed candidate set differs')
                    for base in c.NOISE[a.split]:
                        noise=base+j; addr=(r['sha256'],s,noise)
                        if addr in index:continue
                        chosen=choice['candidate']
                        pending.append((s,noise,chosen,dict(row=r,plan=chosen['plan'],interface=chosen['interface'],mappo_seed=s,action_seed=noise)))
                for (s,noise,chosen,_),metric in zip(pending,pool.map(R.worker,[t for _,_,_,t in pending])):
                    rec=dict(config_sha256=ch,llm_seed=a.seed,mappo_seed=s,instance_key=r['key'],instance_sha256=r['sha256'],
                        ordinal=j,action_seed=noise,K=a.K,actual_K=len(candidate),interface=chosen['interface'],invalid=not raw['check']['valid'],
                        selected=chosen['id'],metrics=metric)
                    c.append(out/'results.jsonl',rec); index[r['sha256'],s,noise]=rec
                print('EVAL',a.split,'seed',a.seed,'J',a.J,'instance',j,flush=True)
        for s,proof in proofs.items():
            if c.sha(Path(proof['run'])/'checkpoints/latest.pt')!=proof['checkpoint_sha256']:
                raise ValueError('checkpoint changed during evaluation')
        if set(index)!=expected:raise ValueError('evaluation grid incomplete')
        c.write_json(out/'complete.json',dict(complete=True,config_sha256=ch,rows=len(index),
            invalid_instances=sum(not indexed[r['sha256']]['check']['valid'] for _,r in selected),
            invalid_episodes=sum(r['invalid'] for r in index.values()),results_sha256=c.sha(out/'results.jsonl')))
        print('EVAL_DONE',a.split,'seed',a.seed,'J',a.J,'episodes',len(index),flush=True)


if __name__=='__main__':main()
