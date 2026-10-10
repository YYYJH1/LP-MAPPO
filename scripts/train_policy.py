"""Train a MARL policy: MAPPO, which LP-MAPPO uses, or one of the baselines IPPO, HAPPO, CoPPO and A2PO.

  CUDA_VISIBLE_DEVICES=0 python scripts/train_policy.py --algorithm mappo --seed 42 --output-dir data/checkpoints/S2_mappo_s42

Training uses the train and monitor splits in data/instances, 192 iterations of 64 episodes, one GPU and 12 CPU
workers. scripts/evaluate.py reads the MAPPO policies from data/checkpoints/S2_mappo_s<seed>.
"""
import argparse
import os
from pathlib import Path
import sys

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import lawn_mec.marl_v2.train as driver

DEFAULTS = ['--seed', '42', '--pool-manifest', str(REPO/'data/instances/splits/train.json'),
            '--eval-pool-manifest', str(REPO/'data/instances/splits/monitor.json'),
            '--episodes-per-iteration', '64', '--iterations', '192', '--workers', '12', '--device', 'cuda',
            '--execution-layer', 'audit', '--f1', '--f2', '--f3', '--f4', '--f5', '--f6', '--f7', '--f8', '--f9',
            '--f14a', '--no-f14b', '--no-f14c', '--f15', '--f16']
_parser = driver.parser


def parser():
    ap = _parser()
    ap.description, ap.formatter_class = __doc__, argparse.RawDescriptionHelpFormatter
    return ap


def main():
    argv = DEFAULTS + sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--algorithm'); pre.add_argument('--seed'); pre.add_argument('--output-dir')
    known, _ = pre.parse_known_args(argv)
    if known.output_dir is None and known.algorithm:
        argv += ['--output-dir', str(REPO/'runs/train'/f'{known.algorithm}_s{known.seed}')]
    driver.parser = parser
    driver.main(argv)


if __name__ == '__main__':
    main()
