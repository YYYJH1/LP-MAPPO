import argparse, json, multiprocessing as mp, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from lawn_mec.marl_v2.algos import Trainer
from lawn_mec.marl_v2.flags import Flags
from lawn_mec.marl_v2.pool import load_manifest
from lawn_mec.marl_v2.train import common_job, worker, worker_init


def trained(run):
    run = Path(run)
    if not (run/'complete.json').exists():
        raise ValueError(f'{run}: training not complete (no complete.json)')
    meta = json.loads((run/'run.json').read_text())
    ck = torch.load(run/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    if ck['run'].get('sources') != meta.get('sources'):
        raise ValueError(f'{run}: checkpoint provenance differs from run.json')
    trainer = Trainer(meta['algorithm'], meta['state_dim'], meta['U'], device='cpu', seed=meta['seed'],
                      flags=Flags(**meta['flags']))
    trainer.restore(ck)
    return meta, ck['iteration'], common_job(trainer, meta['execution_layer'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--action-seeds', type=int, default=2)
    ap.add_argument('--seed-base', type=int, default=910000)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) != '1':
            ap.error(f'{key}=1 required (rollout workers are single-threaded)')
    manifest = load_manifest(a.manifest)
    rows = manifest['instances']
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add((r['run'], r['instance_sha256'], r['action_seed']))
    jobs = []
    for run in a.runs:
        meta, iteration, common = trained(run)
        for j, row in enumerate(rows):
            for s in range(a.action_seeds):
                seed = a.seed_base + 1000*s + j
                if (str(Path(run).resolve()), row['sha256'], seed) in done:
                    continue
                jobs.append((dict(run=str(Path(run).resolve()), algorithm=meta['algorithm'], train_seed=meta['seed'],
                                  flags={k: v for k, v in meta['flags'].items()
                                         if k.startswith('f14') or k == 'f2' or (k in ('f15', 'f16') and v)},
                                  checkpoint_iteration=iteration, manifest=manifest['manifest_path'],
                                  manifest_sha256=manifest['manifest_sha256'], instance_sha256=row['sha256'],
                                  instance_seed=row.get('seed'), action_seed=seed),
                             dict(common, row=row, seed=seed)))
    print(json.dumps(dict(runs=len(a.runs), instances=len(rows), pending=len(jobs), done=len(done))), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=worker_init) as pool, \
            out.open('a') as stream:
        futures = {pool.submit(worker, job): head for head, job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            head = futures[fut]; m = fut.result().result['metrics']
            stream.write(json.dumps(dict(head, **{k: m[k] for k in ('C', 'K', 'c_min', 'feasible', 'on_time', 'loss',
                                                                     'energy_J', 'eta_J', 'upsilon')},
                                         reward=-m['loss']), allow_nan=False) + '\n')
            stream.flush()
            if n % 50 == 0:
                print(f'{n}/{len(jobs)}', flush=True)


if __name__ == '__main__':
    main()
