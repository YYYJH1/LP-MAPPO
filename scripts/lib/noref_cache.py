import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import time
import noref_common as c


def build_one(r):
    c.runtime()
    from lawn_mec.llm import vcp_grammar as vg
    ctx = vg.InstanceContext(vg._identity(r['path']), 'nominal')
    tic = time.perf_counter(); ctx.dp, ctx.reference
    target = c.cache_paths(vg._identity(r['path']), r['sha256'])[0]
    if not target.is_relative_to(c.TASK/'runtime'):
        raise ValueError('new contexts must be stored below the runs directory')
    vg._save(target, ctx)
    if not target.is_file():
        raise RuntimeError('cache save failed')
    return dict(key=r['key'], seconds=time.perf_counter()-tic, path=str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=c.EXPECTED, required=True)
    p.add_argument('--build', action='store_true'); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--out', help='default: <runs-root>/cache/<split>')
    c.add_path_args(p); a = c.configure(p.parse_args()); a.out = a.out or str(c.TASK/'cache'/a.split)
    if not 1 <= a.workers <= 32:
        p.error('workers must be 1..32')
    c.runtime(); m = c.manifest(a.split)
    rr = m['instances']; missing = [r for r in rr if c.cached_path(r) is None]
    out = c.output(Path(a.out)/'inventory.json').parent
    c.pin(out/'config.json', dict(split=a.split, manifest_sha256=m['manifest_sha256'], code_version=c.VERSION))
    tic = time.perf_counter(); built = []
    if a.build and missing:
        os.environ['LAWN_CONTEXT_CACHE'] = str(c.TASK/'runtime/context_store')
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            for row in pool.map(build_one, missing):
                c.append(out/'built.jsonl', row); built.append(row); print(row, flush=True)
    absent = [r['key'] for r in rr if c.cached_path(r) is None]
    result = dict(split=a.split, instances=len(rr), available=len(rr)-len(absent), missing=absent,
                  built=len(built), wall_s=time.perf_counter()-tic, code_version=c.VERSION, reference_rate='nominal')
    c.write_json(out/'inventory.json', result); print('NOREF_CACHE', result['instances'], result['available'], len(absent), flush=True)


if __name__ == '__main__':
    main()
