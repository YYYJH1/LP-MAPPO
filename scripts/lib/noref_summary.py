import numpy as np
import noref_common as c

METRICS=('eta_kJ','C','on_time','feasible','energy_kJ')


def panel(records,keys,split):
    index={(r['llm_seed'],r['mappo_seed'],r['instance_sha256'],r['action_seed']):r['metrics'] for r in records}
    expected={(s,m,k,b+j) for s in c.SEEDS for m in c.MAPPOS for j,k in enumerate(keys) for b in c.NOISE[split]}
    if len(index)!=len(records) or set(index)!=expected:
        raise ValueError('incomplete/duplicate paired LLM/MAPPO/instance/noise grid')
    a=np.array([[[[np.mean([index[s,m,k,b+j][f] for b in c.NOISE[split]])
         for f in ('energy_J','C','K','feasible')] for j,k in enumerate(keys)] for m in c.MAPPOS] for s in c.SEEDS],dtype=float)
    if not np.isfinite(a).all():raise ValueError('nonfinite metric')
    return a


def statistics(a):
    e,cc,k=a[:,:,:,0].mean(axis=2),a[:,:,:,1].mean(axis=2),a[:,:,:,2].mean(axis=2)
    if np.any(cc<=0) or np.any(k<=0):raise ValueError('undefined completion/eta ratio')
    per_m=np.stack((e/cc/1000,cc,cc/k,a[:,:,:,3].mean(axis=2),e/1000),axis=2)
    per_seed=per_m.mean(axis=1)
    return per_seed,per_seed.mean(axis=0),per_m


def compare(a,b,reps=10000):
    aps,am,amp=statistics(a); bps,bm,bmp=statistics(b)
    rng=np.random.default_rng(20261002); boot=[]; seedboot=[]
    for _ in range(reps):
        ids=rng.integers(a.shape[2],size=a.shape[2]); av,aa,_=statistics(a[:,:,ids]); bv,bb,_=statistics(b[:,:,ids])
        boot.append([100*(aa[0]/bb[0]-1),*(aa-bb)]); seedboot.append(av-bv)
    ci=np.quantile(np.asarray(boot),[.025,.975],axis=0); sci=np.quantile(np.asarray(seedboot),[.025,.975],axis=0)
    return dict(candidate=dict(zip(METRICS,map(float,am))),baseline=dict(zip(METRICS,map(float,bm))),
        delta=dict(zip(METRICS,map(float,am-bm))),CI95_delta={f:ci[:,i+1].tolist() for i,f in enumerate(METRICS)},
        eta_delta_percent=float(100*(am[0]/bm[0]-1)),CI95_eta_delta_percent=ci[:,0].tolist(),
        by_seed={str(s):dict(candidate=dict(zip(METRICS,map(float,aps[i]))),baseline=dict(zip(METRICS,map(float,bps[i]))),
          eta_ratio=float(aps[i,0]/bps[i,0]),delta=dict(zip(METRICS,map(float,aps[i]-bps[i]))),
          CI95_delta={f:sci[:,i,j].tolist() for j,f in enumerate(METRICS)},
          by_mappo={str(m):dict(candidate=dict(zip(METRICS,map(float,amp[i,j]))),baseline=dict(zip(METRICS,map(float,bmp[i,j])))) for j,m in enumerate(c.MAPPOS)}) for i,s in enumerate(c.SEEDS)},
        bootstrap_reps=reps,bootstrap_seed=20261002,
        paired_unit='instance; all three LLM seeds, five MAPPO seeds, both noises resampled together',
        aggregation='per MAPPO sum(E)/sum(C), sum(C)/sum(K); then mean over MAPPO and LLM seeds')
