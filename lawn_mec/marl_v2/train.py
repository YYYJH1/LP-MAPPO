import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import torch

from .flags import Flags, flags_record
from .pool import load_manifest, filter_witnessed
from .env import MarlEnv
from .nets import ActorBank, Critic, ValueNorm
from .algos import Trainer, Config, DISPLAY_NAMES
from .rollout import collect, ExecutorCache
from .batch import Batch
from lawn_mec.exec_v2.events import EventExecutor

_cache = None


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_digest():
    root = Path(__file__).parent.parent
    return {name: hashlib.sha256(b''.join(
        str(p.relative_to(root)).encode()+b'\0'+p.read_bytes()
        for p in sorted((root/name).rglob('*.py')))).hexdigest() for name in ('marl_v2','exec_v2')}


EXECUTOR_CACHE = 1
LEGACY_EXECUTOR_CACHE = 16


def worker_init(executor_cache=LEGACY_EXECUTOR_CACHE):
    global _cache
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) not in ('1', '8'):
            raise RuntimeError(f'{key} must be 1 or 8 for rollouts')
    if os.getpriority(os.PRIO_PROCESS,0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    torch.set_num_threads(1)
    _cache = ExecutorCache(capacity=executor_cache)


def worker(job):
    flags = Flags(**job['flags'])
    actors = ActorBank(job['U'],job['shared'],flags=flags)
    critic = Critic(job['state_dim'],job['U'],job['local'],flags=flags)
    norm = ValueNorm()
    if flags.f14c:
        from .f14_value import PotentialResidualCritic
        critic = PotentialResidualCritic(critic,norm)
    actors.load_state_dict(job['actors']); critic.load_state_dict(job['critic'])
    norm.load_state_dict(job['normalizer'])
    instance = json.loads(Path(job['row']['path']).read_text())
    start = time.perf_counter()
    skip = job.get('skip_terminal_replay', False)
    executor = _cache.fresh(instance, execution_layer=job['execution_layer'], terminal_replay=not skip)
    if skip:
        from .audit_execution import collect as audit_collect
        episode = audit_collect(instance,actors,critic,norm,seed=job['seed'],executor=executor,
                                flags=flags,skip_terminal_replay=True)
    else:
        extra = {}
        if job.get('pace', 'off') != 'off':
            from lawn_mec.llm import vcp_grammar as vg
            if vg._context.cache_info().maxsize != 1:
                vg.set_context_capacity(1)
            extra = dict(pace=job['pace'], pace_context=vg.get_context(job['row']['key'], job['pace_registry'],
                                                                 reference_rate=job.get('reference_rate', 'nominal')))
        episode = collect(instance,actors,critic,norm,seed=job['seed'],executor=executor,flags=flags,**extra)
    episode.wall_s = time.perf_counter()-start
    episode.result['instance_sha256'] = job['row']['sha256']
    if job.get('artifact'):
        path = Path(job['artifact']); path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as stream: json.dump(episode.result,stream,allow_nan=False)
    return episode


def common_job(trainer, execution_layer='evaluation', *, skip_terminal_replay=False):
    return dict(U=trainer.actors.uavs,shared=trainer.actors.shared,local=trainer.critic.local,
                state_dim=trainer.critic.state_dim,flags=asdict(trainer.flags),execution_layer=execution_layer,
                skip_terminal_replay=skip_terminal_replay,
                **{name:{k:v.detach().cpu().clone() for k,v in getattr(trainer,name).state_dict().items()}
                   for name in ('actors','critic','normalizer')})


def save_checkpoint(path, trainer, iteration, run):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(dict(trainer.checkpoint(), iteration=iteration, run=run), temp)
    temp.replace(path)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--algorithm',choices=DISPLAY_NAMES,required=True)
    ap.add_argument('--seed',type=int,default=200)
    ap.add_argument('--pool-manifest',required=True)
    ap.add_argument('--eval-pool-manifest')
    ap.add_argument('--episodes-per-iteration',type=int)
    ap.add_argument('--iterations',type=int,default=40)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    ap.add_argument('--execution-layer',default='evaluation',choices=('evaluation','audit'),
                    help='F12 executor mode (default: evaluation)')
    ap.add_argument('--skip-terminal-replay',action='store_true',
                    help='F12 audit-mode training only; paired evaluation always replays')
    ap.add_argument('--executor-cache',type=int,default=EXECUTOR_CACHE,
                    help='executor templates cached per rollout worker')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--resume',nargs='?',const='auto')
    ap.add_argument('--prepare-only',action='store_true')
    for name, default in asdict(Flags()).items():
        ap.add_argument('--'+name, action=argparse.BooleanOptionalAction,default=default)
    return ap


def main(argv=None):
    ap = parser(); args = ap.parse_args(argv)
    if not 1 <= args.workers <= 31: ap.error('workers must be 1..31 (plus one trainer process)')
    if args.executor_cache < 1: ap.error('--executor-cache must be positive')
    if args.seed < 0 or args.iterations < 1: ap.error('seed >= 0 and iterations >= 1 required')
    if args.execution_layer == 'audit' and 'execution_layer' not in inspect.signature(EventExecutor).parameters:
        ap.error('this executor does not expose F12 audit execution')
    if args.skip_terminal_replay:
        if args.execution_layer != 'audit': ap.error('--skip-terminal-replay requires --execution-layer audit')
        if 'terminal_replay' not in inspect.signature(EventExecutor).parameters:
            ap.error('this executor does not expose terminal replay selection')
    flags = Flags(**{name:getattr(args,name) for name in asdict(Flags())})
    episodes_n = ((64 if flags.f9 else 32) if args.episodes_per_iteration is None
                  else args.episodes_per_iteration)
    if episodes_n < 1: ap.error('episodes-per-iteration must be positive')
    output = Path(args.output_dir).resolve()
    checkpoint_path = output/'checkpoints/latest.pt'
    if checkpoint_path.exists() and not args.resume: ap.error('existing checkpoint: use --resume or a new output directory')
    raw = load_manifest(args.pool_manifest); pool = filter_witnessed(raw,flags.f5)
    if not pool['instances']: ap.error('no training instances survive witness filtering')
    ev = load_manifest(args.eval_pool_manifest) if args.eval_pool_manifest else raw
    if len(ev['instances']) < 16 and flags.f8: ap.error('F8 requires at least 16 evaluation instances')
    eval_rows = ev['instances'][:16]
    instances = [json.loads(Path(r['path']).read_text()) for r in [*pool['instances'],*eval_rows]]
    U = instances[0]['params']['U']
    if any(x['params']['U'] != U for x in instances): ap.error('one run requires a fixed U; schema meaning supports U=1..6')
    if flags.f4:
        from .state_schema import STATE_DIM, MAX_U, MAX_K, MAX_M, MAX_NODE, MAX_BUILDING
        for x in instances:
            if any(a>b for a,b in ((U,MAX_U),(len(x['tasks']),MAX_K),(x['params']['M'],MAX_M),
                                  (len(x['nodes']),MAX_NODE),(len(x['buildings']),MAX_BUILDING))):
                ap.error('instance exceeds F4 schema v1 maxima')
        state_dim = STATE_DIM
    else:
        state_dim = max(len(MarlEnv(x,flags=flags).full_state()) for x in instances)
    run = dict(algorithm=args.algorithm,seed=args.seed,flags=flags_record(flags),
               pool_sha256=raw['manifest_sha256'],eval_pool_sha256=ev['manifest_sha256'],
               evaluation_scope='external' if args.eval_pool_manifest else 'training_pool_monitor',
               evaluation_instances=[r['sha256'] for r in eval_rows],
               episodes_per_iteration=episodes_n,state_dim=state_dim,U=U,sources=source_digest(),
               execution_layer=args.execution_layer,terminal_replay=not args.skip_terminal_replay,
               evaluation_terminal_replay=True,
               certified_service_available='certified_service' in inspect.signature(EventExecutor).parameters,
               executor_cache=args.executor_cache)
    if args.prepare_only:
        write(output/'filtered_pool.json',pool); write(output/'run.json',run)
        print(json.dumps(dict(prepared=True,filter=pool['filter'],run=run)),flush=True); return
    if args.device=='cuda' and not torch.cuda.is_available():
        ap.error('assigned CUDA device unavailable; no CPU substitution')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) != '1': ap.error(f'{key}=1 required')
    if os.getpriority(os.PRIO_PROCESS,0) < 10: os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    torch.set_num_threads(1)
    trainer = Trainer(args.algorithm,state_dim,U,device=args.device,seed=args.seed,flags=flags)
    start = 0
    if args.resume:
        source = checkpoint_path if args.resume=='auto' else Path(args.resume)
        ck = torch.load(source,map_location='cpu',weights_only=False)
        if ck['run'] != run: ap.error('resume provenance differs (pool/evaluation/flags/source/seed/batch)')
        trainer.restore(ck); start = ck['iteration']+1
        if start > args.iterations: ap.error('requested iterations precede the saved checkpoint')
    write(output/'filtered_pool.json',pool); write(output/'run.json',dict(run,description=trainer.description()))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init,
                             initargs=(args.executor_cache,)) as workers:
        for iteration in range(start,args.iterations):
            if source_digest() != run['sources']:
                raise RuntimeError('source files changed during run; stop before mixing executor versions')
            tic = time.perf_counter(); common = common_job(trainer,args.execution_layer,
                                                        skip_terminal_replay=args.skip_terminal_replay)
            jobs = [dict(common,row=pool['instances'][(iteration*episodes_n+j)%len(pool['instances'])],
                         seed=args.seed+iteration*episodes_n+j,
                         artifact=str(output/'episodes'/f'{iteration:04d}_{j:03d}.json.gz')) for j in range(episodes_n)]
            episodes = list(workers.map(worker,jobs)); rollout_s = time.perf_counter()-tic
            if source_digest() != run['sources']:
                raise RuntimeError('source files changed during collection; discard this batch')
            tic = time.perf_counter(); update = trainer.update(Batch(episodes)); update_s = time.perf_counter()-tic
            row = dict(iteration=iteration,episodes_total=(iteration+1)*episodes_n,rollout_s=rollout_s,
                       update_s=update_s,update=update,
                       on_time=sum(e.result['metrics']['C'] for e in episodes)/sum(e.result['metrics']['K'] for e in episodes),
                       feasible=float(np.mean([e.result['metrics']['feasible'] for e in episodes])))
            if flags.f8 and (iteration+1)%10 == 0:
                common = common_job(trainer,args.execution_layer)
                evaluation = list(workers.map(worker,[dict(common,row=r,seed=900000+j) for j,r in enumerate(eval_rows)]))
                row['evaluation'] = [dict(instance_sha256=r['sha256'],seed=900000+j,**e.result['metrics'])
                                     for j,(r,e) in enumerate(zip(eval_rows,evaluation))]
            write(output/'updates'/f'{iteration:04d}.json',row)
            save_checkpoint(checkpoint_path,trainer,iteration,run)
            curve = [json.loads(p.read_text()) for p in sorted((output/'updates').glob('*.json'))
                     if int(p.stem) <= iteration]
            write(output/'curve.json',curve)
            print(json.dumps(row,allow_nan=False),flush=True)
    curve = [json.loads(p.read_text()) for p in sorted((output/'updates').glob('*.json'))
             if int(p.stem) < args.iterations]
    write(output/'curve.json',curve)
    write(output/'complete.json',dict(iterations=args.iterations,checkpoint=str(checkpoint_path),run=run))


if __name__ == '__main__':
    main()
