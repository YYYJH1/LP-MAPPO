from dataclasses import replace
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import os
import numpy as np

FROZEN_TRAIN = Path(os.environ.get('LPM_DATA_ROOT', Path(__file__).resolve().parents[2]/'data'))/'instances'/'splits'/'train.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_train(path, canonical=FROZEN_TRAIN):
    path=Path(path).resolve(); canonical=Path(canonical).resolve()
    names={x.lower() for x in path.stem.replace('-','_').split('_')}
    if names & {'test','tune','validation','dev','monitor'}:
        raise ValueError('residual probe is train-only; refusing non-train path')
    raw=json.loads(path.read_text())
    if raw.get('split')!='train' or raw.get('sealed'):
        raise ValueError('residual probe requires explicit unsealed train split')
    source=raw if path==canonical else json.loads(canonical.read_text())
    if source.get('split')!='train' or source.get('sealed'):
        raise ValueError('canonical training manifest is not train')
    def identity(row,parent):
        p=Path(row['path']);p=p if p.is_absolute() else parent/p
        return row['key'],row['sha256'],str(p.resolve())
    allowed={identity(r,canonical.parent) for r in source['instances']}
    if any(identity(r,path.parent) not in allowed for r in raw['instances']):
        raise ValueError('subset contains an instance outside canonical train')
    from lawn_mec.marl_v2.pool import load_manifest
    return load_manifest(path)


def enumerate_edits(ctx, *, budget=32, max_tasks=2, upload=False, seed=203):
    if budget<1 or max_tasks not in (1,2):
        raise ValueError('positive edit budget and max_tasks in {1,2} required')
    base=ctx.reference; options={}
    for k,c in enumerate(base.tasks):
        if c.g!='S' or c.s is None or c.a is None:continue
        opts=[]
        for ds,da in product((-1,1),repeat=2):
            if not (0<=c.s+ds<8 and 0<=c.a+da<8):continue
            shifts=(0,-1,1) if upload and c.upload is not None else (0,)
            for du in shifts:
                if c.upload is not None and not 0<=c.upload+du<8:continue
                opts.append(dict(task=k,ds=ds,da=da,du=du))
        if opts:options[k]=opts
    specs=[]
    for count in range(1,max_tasks+1):
        for tasks in combinations(options,count):
            specs.extend(product(*(options[k] for k in tasks)))
    order=np.random.default_rng(seed).permutation(len(specs)); accepted=0
    c2=ctx.dp.precheck(base.keep_sets(ctx.tables))
    if not c2['passed']:
        raise ValueError('DP reference fails C2; no residual labels produced')
    for index in order:
        edit=specs[int(index)];tasks=list(base.tasks)
        for move in edit:
            k=move['task'];c=tasks[k]
            tasks[k]=replace(c,s=c.s+move['ds'],a=c.a+move['da'],
                             upload=None if c.upload is None else c.upload+move['du'])
        plan=replace(base,tasks=tuple(tasks))
        if not ctx.grammar.validate(plan.tokens(ctx.tables))['valid']:continue
        if not ctx.dp.precheck(plan.keep_sets(ctx.tables))['passed']:continue
        yield tuple(edit),plan
        accepted+=1
        if accepted>=budget:return


def improvements(base,edited):
    a=float(np.mean([x['loss'] for x in base]));b=float(np.mean([x['loss'] for x in edited]))
    def eta(rows):
        E=sum(x['energy_J'] for x in rows);C=sum(x['C'] for x in rows)
        return E/C if C else None
    e0,e1=eta(base),eta(edited)
    return dict(L_N_baseline=a,L_N_edit=b,L_N_improvement_percent=100*(a-b)/a,
                eta_baseline_J=e0,eta_edit_J=e1,
                eta_improvement_percent=None if e0 is None or e1 is None else 100*(e0-e1)/e0,
                C_baseline=float(np.mean([x['C'] for x in base])),C_edit=float(np.mean([x['C'] for x in edited])))


def distribution(values):
    valid=[v for v in values if v is not None]
    if not valid:return dict(n=0)
    x=np.asarray(valid,float)
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),min=float(x.min()),
                quantiles=dict(zip(('p05','p25','p50','p75','p95'),map(float,np.quantile(x,[.05,.25,.5,.75,.95])))),
                max=float(x.max()))


def gate(rows, expected, *, reps=10000):
    if reps<1:raise ValueError('bootstrap repetitions must be positive')
    x=np.array([r['holdout']['L_N_improvement_percent'] for r in rows],float)
    lo=hi=None
    if len(x):
        rng=np.random.default_rng(20261002)
        samples=[float(x[rng.integers(len(x),size=len(x))].mean()) for _ in range(reps)]
        lo,hi=map(float,np.quantile(samples,[.025,.975]))
    complete=len(rows)==expected and expected>=256
    criteria=dict(mean_improvement_at_least_0_5_percent=bool(len(x) and x.mean()>=.5),
                  bootstrap_lower_above_zero=bool(lo is not None and lo>0),
                  positive_fraction_at_least_0_2=bool(len(x) and np.mean(x>0)>=.2))
    return dict(status='pass' if complete and all(criteria.values()) else 'fail' if complete else 'incomplete',
                instances=len(rows),expected=expected,minimum_instances=256,criteria=criteria,
                mean_improvement_percent=float(x.mean()) if len(x) else None,
                CI95_percent=[lo,hi],positive_fraction=float(np.mean(x>0)) if len(x) else None,
                bootstrap_reps=reps,paired_unit='instance; the two holdout repetitions remain together',
                screening={k:distribution([r['screening'][k] for r in rows]) for k in ('L_N_improvement_percent','eta_improvement_percent')},
                holdout={k:distribution([r['holdout'][k] for r in rows]) for k in ('L_N_improvement_percent','eta_improvement_percent')},
                sft_greedy_non_dp='pending SFT; this probe cannot verify greedy model outputs')
