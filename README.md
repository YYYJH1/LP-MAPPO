# LP-MAPPO

LLM-guided plan-constrained multi-agent PPO for cooperative UAV edge computing in urban air corridors.

The repository contains the simulator and instance generator, the MARL policies (MAPPO and the IPPO, HAPPO, CoPPO
and A2PO baselines), LoRA fine-tuning and plan generation for the LLM planner, the plan-constrained evaluation and
the evaluation data.

```text
lawn_mec/   simulator, instance generator, MARL policies, plan grammar and DP reference planner, LLM planner
scripts/    command-line entry points (shared code in scripts/lib)
data/       scenario parameters and evaluation data (data/README.md)
```

## Installation

Python 3.12 on Linux:

```bash
pip install -r requirements.txt            # simulation, policy training and evaluation
pip install -r requirements-planner.txt    # planner fine-tuning and plan generation (GPU)
```

The planner is fine-tuned from Qwen3-1.7B (Hugging Face `Qwen/Qwen3-1.7B`), expected in `models/qwen3-1.7b`. The
fine-tuned LoRA adapters are attached to release v2.0: `tar -xf lp-mappo-lora-s42.tar` in the repository root
creates `adapters/s42/`.

## Usage

Run the commands from the repository root. Every script lists its options with `--help`.

```bash
# instance sets (CPU)
python scripts/make_instances.py --split train --count 1747 --workers 64 --out data/instances
for s in tune monitor validation test; do
  python scripts/make_instances.py --split $s --workers 64 --out data/instances
done

# MAPPO policies (one GPU per run); the baselines use --algorithm ippo, happo, coppo or a2po
for s in 42 43 44 45 46; do
  python scripts/train_policy.py --algorithm mappo --seed $s --output-dir data/checkpoints/S2_mappo_s$s
done

# planner fine-tuning
python scripts/train_planner.py --prepare-only --workers 32
for s in 42 43 44; do python scripts/train_planner.py --seed $s --gpu 0; done

# plan generation and evaluation
for s in 42 43 44; do
  python scripts/generate_plans.py --seed $s --gpu 0 --adapter runs/planner/sft/s$s/adapter
  python scripts/evaluate.py --seed $s --plans runs/plans/test/s$s/plans.jsonl --workers 10
done
python scripts/summarize.py --records runs/eval/test
```

`summarize.py` compares the recomputed test means with `data/summaries/lp_mappo/test_summary.json`.

## License

MIT ([LICENSE](LICENSE)), except for the following third-party code:

- `lawn_mec/marl_v2/official_mlp.py`, `lawn_mec/marl_v2/official_util.py` and the `ValueNorm` class in
  `lawn_mec/marl_v2/nets.py` are adapted from [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)
  (MIT, `lawn_mec/marl_v2/LICENSE.on-policy`).
- The A2PO components of `lawn_mec/marl_v2/algos/ppo.py` are adapted from
  [xihuai18/A2PO-ICLR2023](https://github.com/xihuai18/A2PO-ICLR2023) (MIT, `lawn_mec/marl_v2/LICENSE.A2PO`). The
  CoPPO baseline follows the CoPPO option of that repository.
- The MAPPO, HAPPO and HATRPO hyperparameters follow the configuration files of
  [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL).
- `lawn_mec/vcpm` keeps the licence texts of [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent),
  [OpenMLRL/CoMLRL](https://github.com/OpenMLRL/CoMLRL) and marlbenchmark/on-policy, which were consulted for it.
