"""Generate and certify simulation instances.

Each instance is generated from data/instances/params.json and a seed, then certified by a time-limited offline
search for a feasible schedule, which also sets the battery capacity. Seeds without a feasible schedule are skipped.

  python scripts/make_instances.py --split test --workers 64 --out data/instances
  python scripts/make_instances.py --split train --count 1747 --workers 64 --out data/instances
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SPLITS = {
    'train': (1100000, 2048, 0, 256), 'tune': (1170000, 128, 512, 64), 'monitor': (1200000, 64, 512, 0),
    'validation': (1300000, 128, 512, 128), 'test': (1400000, 256, 512, 0)}
SEARCH = dict(lsc_evaluations=64, lsc_cpu_s=240., el_attempts=10000, execution_layer='audit')
MARGIN = 0.1
PARAMS_SHA256 = 'a86cc1aeea1ed11a2b83aa8f46bfd144f168325cea1b01a33dcd76c305d23683'


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part'); part.write_bytes(encoded(value)); part.replace(path)


def fingerprints(instance):
    params = {k: v for k, v in instance['params'].items() if not k.startswith('battery')}
    fields = ('tasks', 'visits', 'nodes', 'edges', 'buildings', 'no_fly', 'bs', 'load', 'qI', 'qF', 'v0')
    return dict(content_sha256=digest(instance), physics_sha256=digest(dict(params=params, **{k: instance[k] for k in fields})))


def generate_one(job):
    split, seed, params, out = job
    from lawn_mec.env.generator import generate
    path = Path(out)/'inputs'/f'{split}_{seed}.json'
    if not path.exists():
        write(path, generate(params, seed))
    return dict(seed=seed, sha256=sha(path), **fingerprints(json.loads(path.read_text())))


def certify_one(job):
    split, seed, out, budget = job
    from lawn_mec.vcp.certify import search, anchor, serializable
    key, out, tic = f'{split}_{seed}', Path(out), time.process_time()
    source = out/'inputs'/f'{key}.json'; instance = json.loads(source.read_text())
    result = search(instance, budget, search_seed=seed, **SEARCH)
    attempts = [{k: result.get(k) for k in ('budget_s', 'cpu_s', 'status', 'stop', 'completed_evaluations')}]
    if result['status'] != 'witness' and split != 'train':
        result = search(instance, 2*budget, search_seed=seed, **SEARCH)
        attempts.append({k: result.get(k) for k in ('budget_s', 'cpu_s', 'status', 'stop', 'completed_evaluations')})
    final = serializable(anchor(instance, result, MARGIN))
    if fingerprints(final)['physics_sha256'] != fingerprints(instance)['physics_sha256']:
        raise AssertionError('certification changed the scenario physics of ' + key)
    write(out/'certified'/f'{key}.json', final)
    record = dict(key=key, seed=seed, status=final['certification']['status'], input_sha256=sha(source),
                  certification=final['certification'], attempts=serializable(attempts),
                  planner=serializable({k: v for k, v in result.items() if k not in ('incumbent', 'trace', 'lsc_trace')}),
                  witness=None, worker_cpu_s=time.process_time()-tic)
    if result['incumbent'] is not None:
        witness = out/'witnesses'/f'{key}.json.gz'; witness.parent.mkdir(parents=True, exist_ok=True)
        witness.write_bytes(gzip.compress(encoded(serializable(result['incumbent'])), mtime=0))
        record['witness'] = dict(path=f'witnesses/{key}.json.gz', sha256=sha(witness))
    write(out/'certification'/f'{key}.json', record)
    return record


def certify_in_new_process(job):
    with ProcessPoolExecutor(1, mp_context=mp.get_context('spawn')) as pool:
        return pool.submit(certify_one, job).result()


def check_inputs(a, params, out, pool):
    released = REPO/'data/instances/splits'/f'{a.split}.json'
    if not released.is_file():
        raise SystemExit(f'{released} not found; --check-inputs compares with an existing split manifest')
    rows = json.loads(released.read_text())['instances'][:a.count]
    made = list(pool.map(generate_one, [(a.split, r['seed'], params, str(out)) for r in rows]))
    bad = [r['key'] for r, g in zip(rows, made) if (g['sha256'], g['physics_sha256']) != (r['input_sha256'], r['physics_sha256'])]
    print(f'{len(rows) - len(bad)} of {len(rows)} generated {a.split} inputs equal the manifest hashes', flush=True)
    if bad:
        raise SystemExit(f'FAIL: {bad[:5]}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=SPLITS, required=True)
    p.add_argument('--count', type=int, help='number of instances (default: split size; train: all witnessed seeds)')
    p.add_argument('--workers', type=int, default=4, help='parallel processes')
    p.add_argument('--budget', type=float, default=600., help='CPU seconds of a witness search (default 600)')
    p.add_argument('--params', default=str(REPO/'data/instances/params.json'), help='scenario parameters')
    p.add_argument('--out', default=str(REPO/'runs/instances'), help='output directory')
    p.add_argument('--check-inputs', action='store_true',
                   help='only regenerate the inputs listed in data/instances/splits/<split>.json and compare hashes')
    a = p.parse_args()
    if a.count is not None and a.count < 1 or a.workers < 1 or a.budget <= 0:
        p.error('--count, --workers and --budget must be positive')
    params, out = json.loads(Path(a.params).read_text()), Path(a.out).resolve()
    print('scenario parameters', 'equal' if digest(params) == PARAMS_SHA256 else 'differ from', 'the reference parameter set')
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(out/'numba'))
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    spawn = mp.get_context('spawn')
    if a.check_inputs:
        with ProcessPoolExecutor(a.workers, mp_context=spawn) as pool:
            return check_inputs(a, params, out, pool)
    start, primary, extra, skipped = SPLITS[a.split]
    order = list(range(start + skipped, start + primary + extra))
    target = a.count or (None if a.split == 'train' else primary)
    records = {}
    for s in order:
        f = out/'certification'/f'{a.split}_{s}.json'
        if f.is_file():
            records[s] = json.loads(f.read_text())
    while True:
        chosen = []
        for s in order:
            if s not in records or (target and len(chosen) == target):
                break
            if records[s]['status'] == 'witness':
                chosen.append(s)
        todo = [s for s in order if s not in records]
        if (target and len(chosen) == target) or not todo:
            break
        batch = todo[:target - len(chosen) if target else len(todo)]
        print(f'{a.split}: {len(chosen)} witnessed so far; certifying seeds {batch[0]}-{batch[-1]} ({len(batch)})', flush=True)
        with ProcessPoolExecutor(a.workers, mp_context=spawn) as pool:
            inputs = {g['seed']: g for g in pool.map(generate_one, [(a.split, s, params, str(out)) for s in batch])}
        with ThreadPoolExecutor(a.workers) as threads:
            for record in threads.map(certify_in_new_process, [(a.split, s, str(out), a.budget) for s in batch]):
                records[record['seed']] = record
                print(f"  {record['key']}: {record['status']} (input {inputs[record['seed']]['sha256'][:12]})", flush=True)
    if target and len(chosen) < target:
        raise SystemExit(f'only {len(chosen)} of {target} seeds have a witness; no manifest written')
    entries = []
    for s in chosen:
        key = f'{a.split}_{s}'; record = records[s]; path = out/'certified'/f'{key}.json'
        entries.append(dict(seed=s, key=key, split=a.split, path=f'../certified/{key}.json', sha256=sha(path),
                            input_sha256=record['input_sha256'], **fingerprints(json.loads(path.read_text())),
                            witness_battery_J=record['certification']['battery_J'], certification=record['certification'],
                            certification_record=dict(path=f'../certification/{key}.json', sha256=sha(out/'certification'/f'{key}.json')),
                            witness=dict(record['witness'], path='../'+record['witness']['path'])))
    walked = [s for s in order if s <= chosen[-1]] if chosen else []
    excluded = ([dict(seed=s, reason='duplicate') for s in range(start, start + skipped)]
                + [dict(seed=s, reason='no_witness') for s in walked if records[s]['status'] != 'witness'])
    manifest = dict(schema='vcp-freeze-split-v1', pool_name=a.split, split=a.split, status='frozen', sealed=a.split == 'test',
                    execution_mode='audit', target_size=primary, instances=entries, excluded=excluded,
                    added_reserve_seeds=[s for s in chosen if s >= start + primary],
                    replaced_primary_seeds=[e['seed'] for e in excluded if e['seed'] < start + primary],
                    selection='increasing seeds; deduplication then witness status only', params_sha256=digest(params))
    write(out/'splits'/f'{a.split}.json', manifest)
    print(f'{a.split}: {len(entries)} certified instances, manifest {out}/splits/{a.split}.json', flush=True)


if __name__ == '__main__':
    main()
