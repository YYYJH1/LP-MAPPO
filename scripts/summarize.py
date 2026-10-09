"""Summarize the LP-MAPPO test records.

For each combination of planner seed and MAPPO seed, energy per on-time task is the total fleet energy divided by the
total on-time count. The summary reports means and standard errors over the 15 combinations.

  python scripts/summarize.py --records runs/eval/test

When data/summaries/lp_mappo/test_summary.json exists, the recomputed means are compared with it.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'lib'))
import noref_common as c

COLUMNS = (('eta_kJ', 'eta (kJ)'), ('C', 'C^c[N]'), ('on_time', 'On-time'), ('feasible', 'Feasible'),
           ('energy_kJ', 'Energy (kJ)'), ('loss', 'L_N'), ('upsilon', 'upsilon[N]'), ('reward', 'Reward'))
LABEL = {1: 'LP-MAPPO, J=0', 10: 'LP-MAPPO + plan verification (optional), J=10'}


def load(root, K, m):
    from lawn_mec.marl_v2.train import source_digest
    data, proof, seeds = [], [], []
    for s in c.SEEDS:
        paths = sorted((Path(root)/f'K{K}/s{s}').rglob('results.jsonl'))
        if paths:
            seeds.append(s)
        for path in paths:
            cfg = json.loads(path.with_name('config.json').read_text()); done = json.loads(path.with_name('complete.json').read_text())
            if (not done['complete'] or done['results_sha256'] != c.sha(path) or done['config_sha256'] != c.digest(cfg)
                    or cfg['seed'] != s or cfg['split'] != 'test' or cfg['K'] != K or cfg['manifest_sha256'] != m['manifest_sha256']
                    or cfg['code_version'] != c.VERSION or cfg['noise_bases'] != list(c.NOISE['test'])):
                raise ValueError('test record/config/completion differs: '+str(path))
            for ck in cfg['checkpoints'].values():
                if (ck['evaluated_sources'] != source_digest()
                        or c.sha(c.RUNS/Path(ck['run']).name/'checkpoints/latest.pt') != ck['checkpoint_sha256']):
                    raise ValueError('checkpoint or library source differs from the records: '+str(path))
            rows = c.rows(path)
            if any(r['config_sha256'] != c.digest(cfg) for r in rows):
                raise ValueError('result rows and config differ: '+str(path))
            data.extend(rows); proof.append(dict(path=str(path), sha256=c.sha(path)))
    return data, proof, seeds


def pair_values(rows):
    acc = defaultdict(lambda: defaultdict(float))
    for r in rows:
        x = r['metrics']; a = acc[r['llm_seed'], r['mappo_seed']]
        a['n'] += 1; a['E'] += x['energy_J']; a['C'] += x['C']; a['K'] += x['K']
        a['feasible'] += x['feasible']; a['loss'] += x['loss']; a['upsilon'] += x['upsilon']
    return {key: dict(eta_kJ=a['E']/a['C']/1000, C=a['C']/a['n'], on_time=a['C']/a['K'], feasible=a['feasible']/a['n'],
                      energy_kJ=a['E']/a['n']/1000, loss=a['loss']/a['n'], upsilon=a['upsilon']/a['n'],
                      reward=-a['loss']/a['n'], episodes=int(a['n'])) for key, a in sorted(acc.items())}


def mean_se(values):
    n = len(values); mean = sum(values)/n
    return mean, (math.sqrt(sum((v-mean)**2 for v in values)/(n-1)/n) if n > 1 else float('nan'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--records', help='per-episode records with K1/ and K10/ folders (default: <data-root>/records/test)')
    p.add_argument('--out', help='summary output directory (default: <runs-root>/summary/test)')
    p.add_argument('--reps', type=int, default=10000, help='bootstrap draws')
    c.add_path_args(p); a = c.configure(p.parse_args())
    a.records = a.records or str(c.DATA/'records/test'); a.out = a.out or str(c.TASK/'summary/test')
    if a.reps < 1:
        p.error('positive bootstrap draws required')
    c.runtime(); m = c.manifest('test'); keys = [r['sha256'] for r in m['instances']]
    reference = c.DATA/'summaries/lp_mappo/test_summary.json'
    released = json.loads(reference.read_text())['metrics'] if reference.exists() else {}
    data, proof, complete = {}, [], []
    for K in (1, 10):
        rows, files, seeds = load(a.records, K, m)
        if not rows:
            continue
        data[K] = rows; proof += files
        per = pair_values(rows)
        episodes = sum(v['episodes'] for v in per.values())
        full = len(per) == len(c.SEEDS)*len(c.MAPPOS) and episodes == len(per)*len(keys)*2
        if full:
            complete.append(K)
        print(f'{LABEL[K]}: {len(per)} pairs of LoRA and MAPPO seeds, {episodes} test episodes'
              + ('' if full else ' (incomplete: complete records have 15 pairs and 7,680 episodes)'))
        agree = True
        for field, label in COLUMNS:
            mean, se = mean_se([v[field] for v in per.values()])
            note = ''
            if full and field in released.get(str(K), {}):
                same = math.isclose(mean, released[str(K)][field], rel_tol=1e-9, abs_tol=1e-12); agree &= same
                note = '' if same else f'   differs from the reference summary ({released[str(K)][field]:.6f})'
            print(f'  {label:12s} {mean:10.3f} +/- {se:.3f}{note}')
        if full and not released:
            print(f'  no reference summary at {reference}; comparison skipped')
        elif full:
            print('  means ' + ('agree with' if agree else 'DIFFER from') + ' data/summaries/lp_mappo/test_summary.json')
            if not agree:
                raise SystemExit('FAIL: recomputed means differ from the reference summary')
    if not data:
        raise SystemExit(f'no records under {a.records}/K1 or {a.records}/K10')
    if complete != [1, 10]:
        print('Paired J=10 minus J=0 summary skipped: it needs complete J=0 and J=10 records.')
        return
    from noref_summary import panel, statistics, compare, METRICS
    panels = {K: panel(rows, keys, 'test') for K, rows in data.items()}
    metrics = {str(K): dict(zip(METRICS, map(float, statistics(x)[1]))) for K, x in panels.items()}
    result = dict(split='test', instances=256, seeds=list(c.SEEDS), metrics=metrics,
                  K10_minus_K1=compare(panels[10], panels[1], a.reps),
                  invalid_episodes={str(K): sum(r['invalid'] for r in rows) for K, rows in data.items()}, provenance=proof,
                  protocol='final_test_20261003 eval 1010000/1011000; K10 screen 1060000; R/B/L + 7 LLM-base edits')
    c.write_json(Path(a.out)/'summary.json', result)
    d = result['K10_minus_K1']
    print(f"J=10 minus J=0 ({a.reps} instance bootstrap draws): eta {d['delta']['eta_kJ']:+.3f} kJ "
          f"[{d['CI95_delta']['eta_kJ'][0]:+.3f}, {d['CI95_delta']['eta_kJ'][1]:+.3f}], "
          f"C {d['delta']['C']:+.3f} [{d['CI95_delta']['C'][0]:+.3f}, {d['CI95_delta']['C'][1]:+.3f}]")
    print('summary written to', Path(a.out)/'summary.json')


if __name__ == '__main__':
    main()
