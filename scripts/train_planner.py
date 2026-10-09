"""Fine-tune the LLM planner (Qwen3-1.7B with LoRA) on the DP reference plans of the training instances.

  python scripts/train_planner.py --prepare-only --workers 32    # fine-tuning data, CPU
  python scripts/train_planner.py --seed 42 --gpu 0              # LoRA fine-tuning, one GPU

The base model is read from models/qwen3-1.7b (or --model). The adapter is written to runs/planner/sft/s<seed>/adapter.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'lib'))
import noref_common as c


def run(script, *args, env=None):
    command = [sys.executable, str(c.LIB/script), *map(str, args)]
    print('RUN', ' '.join(command[1:]), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, choices=c.SEEDS, help='LoRA seed (42, 43 or 44)')
    p.add_argument('--gpu', type=int, help='physical GPU id used for fine-tuning')
    p.add_argument('--workers', type=int, default=16, help='CPU processes that build instance contexts')
    p.add_argument('--prepare-only', action='store_true', help='build contexts and fine-tuning data, then stop')
    c.add_path_args(p, model=True); a = c.configure(p.parse_args())
    if not a.prepare_only and (a.seed is None or a.gpu is None):
        p.error('--seed and --gpu are required for fine-tuning')
    paths = ['--data-root', a.data_root, '--runs-root', a.runs_root]
    prepared = c.TASK/'planner/data/complete.json'
    if not (prepared.is_file() and json.loads(prepared.read_text()).get('complete')):
        run('noref_cache.py', '--split', 'train', '--build', '--workers', a.workers, *paths)
        run('noref_prepare.py', *paths, '--model', a.model)
    if a.prepare_only:
        return
    run('noref_train.py', '--seed', a.seed, '--gpu', a.gpu, *paths, '--model', a.model,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)))


if __name__ == '__main__':
    main()
