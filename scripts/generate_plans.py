"""Generate coordination plans with a fine-tuned LoRA adapter (vLLM, one GPU).

  python scripts/generate_plans.py --seed 42 --gpu 0                                            # release adapter adapters/s42
  python scripts/generate_plans.py --seed 42 --gpu 0 --adapter runs/planner/sft/s42/adapter

Plans are written to runs/plans/<split>/s<seed>/plans.jsonl.
"""
import argparse
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
    p.add_argument('--seed', type=int, choices=c.SEEDS, required=True, help='LoRA seed of the adapter')
    p.add_argument('--split', choices=('tune', 'validation', 'test'), default='test')
    p.add_argument('--gpu', type=int, help='physical GPU id used for generation')
    p.add_argument('--adapter', help='adapter directory (default: adapters/s<seed>/adapter, next to complete.json)')
    p.add_argument('--convert-only', action='store_true', help='skip generation and convert existing samples')
    p.add_argument('--samples', help='generator output samples.jsonl (default: <runs-root>/generation/<split>/s<seed>/samples.jsonl)')
    c.add_path_args(p, model=True); a = c.configure(p.parse_args())
    samples = Path(a.samples or c.TASK/'generation'/a.split/f's{a.seed}'/'samples.jsonl').resolve()
    paths = ['--data-root', a.data_root, '--runs-root', a.runs_root]
    if not a.convert_only:
        if a.gpu is None:
            p.error('--gpu is required for generation')
        adapter = a.adapter or str(c.CODE/'adapters'/f's{a.seed}'/'adapter')
        run('noref_sample_vllm.py', '--split', a.split, '--seed', a.seed, '--gpu', a.gpu, '--adapter', adapter,
            '--out', samples.parent, *paths, '--model', a.model,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn'))
    run('noref_convert.py', '--split', a.split, '--seed', a.seed, '--source', samples, *paths)


if __name__ == '__main__':
    main()
