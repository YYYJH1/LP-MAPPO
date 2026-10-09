import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from .grammar import CanonicalOptions, DESIGN_OPTIONS, GrammarSpec, encode_trace
from .policy import LoRAPolicy, restricted_logprobs

EVALUATE_AT = ('step', 'ends')


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty SFT dataset')
    for row in rows:
        raw = Path(row['instance_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['instance_sha256']:
            raise ValueError(f"instance hash changed: {row['instance_key']}")
        if not row.get('provenance_sha256'):
            raise ValueError('missing SFT provenance guard')
    evidence = Path(path).parent / 'provenance.json'
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    provenance = json.loads(evidence.read_text())
    from lawn_mec.vcp.payload import require_provenance
    require_provenance(provenance)
    for row in rows:
        require_provenance(row)
    if not provenance.get('passed') or any(r['provenance_sha256'] != digest for r in rows):
        raise ValueError('invalid SFT provenance evidence')
    root = Path(__file__).resolve().parents[3]
    changed = [name for name, expected in provenance.get('source_hashes', {}).items()
               if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
    if changed:
        raise ValueError(f'teacher/executor sources changed; regenerate SFT: {changed}')
    return rows


def traces_for(policy, rows, registry, *, max_length=8192, prompt='compact'):
    if prompt == 'compact':
        from .prompt_compact import build_prompt
    else:
        from .prompt import build_prompt
    spec = GrammarSpec('lawn_mec.llm.vcp_grammar:VCPFieldGrammar', {'registry': str(registry), 'zeta': .85})
    mapping = CanonicalOptions.build(policy.tokenizer, DESIGN_OPTIONS)
    traces = []
    for row in rows:
        key = row['instance_key']
        trace = encode_trace(policy.tokenizer, mapping, spec, key,
                             build_prompt(policy.tokenizer, key, str(registry)), tuple(row['tokens']))
        if spec.build(key).next_field(trace.prefix) is not None:
            raise ValueError('SFT commitment is incomplete')
        if len(trace.token_ids) > max_length:
            raise ValueError(f'{key}: {len(trace.token_ids)} tokens exceeds max_length={max_length}; no truncation')
        traces.append(trace)
    return traces


def autocast_for(model):
    from .precision import autocast
    return autocast(model)


def scaled_step(optimizer, scaler, parameters, backward, *, max_attempts=16, max_grad_norm=1.):
    parameters = list(parameters)
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        value = backward(scaler)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=not scaler.is_enabled())
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
            return value, float(norm), attempt
    raise FloatingPointError('AMP overflow persisted; no update counted')


def field_distributions(policy, trace):
    device = next(policy.model.parameters()).device
    ids = torch.tensor([trace.token_ids], device=device)
    causal = policy.model.get_base_model()
    hidden = causal.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(ids.shape[1], device=device)[None],
                          use_cache=False).last_hidden_state[0]
    head = causal.get_output_embeddings()
    return [(field, torch.log_softmax(
        torch.nn.functional.linear(hidden[field.position-1], head.weight[list(field.allowed_ids)],
                                   None if getattr(head, 'bias', None) is None else head.bias[list(field.allowed_ids)]).float(), -1))
            for field in trace.fields if not field.forced]


def restricted_loss(policy, traces):
    losses = []
    for trace in traces:
        distributions = field_distributions(policy, trace)
        losses.append(-torch.stack([p[f.selected] for f, p in distributions]).sum()
                      if distributions else next(policy.parameters()).sum() * 0)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(policy, traces):
    training = policy.model.training
    policy.model.eval()
    loss, correct, count = 0., 0, 0
    try:
        with autocast_for(policy.model):
            for trace in traces:
                for field, logp in field_distributions(policy, trace):
                    loss -= float(logp[field.selected])
                    correct += int(logp.argmax().item() == field.selected)
                    count += 1
    finally:
        policy.model.train(training)
    return dict(loss=loss/count if count else 0., accuracy=correct/count if count else None,
                active_fields=count, sequences=len(traces), sequence_loss=loss/len(traces))


def train(policy, traces, *, validation=(), steps=3, lr=1e-4, batch_size=8, seed=42,
          checkpoint_dir=None, resume=None, precision='float16', stop_after=None, evaluate_at='step', groups=None):
    if not traces or steps < 1 or batch_size < 1:
        raise ValueError('nonempty traces, positive steps and batch size required')
    if evaluate_at not in EVALUATE_AT:
        raise ValueError(f'evaluate_at must be one of {EVALUATE_AT}')
    if groups is not None:
        import math
        groups = [[[int(i), float(w)] for i, w in members] for members in groups]
        if (sorted(i for members in groups for i, _ in members) != list(range(len(traces)))
                or any(not math.isclose(sum(w for _, w in members), 1., rel_tol=0, abs_tol=1e-9) for members in groups)):
            raise ValueError('groups must partition the traces, with weights summing to 1 per instance')
    from dataclasses import asdict
    from .checkpoint import TrainingState, digest
    from .precision import configure
    configure(policy.model, precision)
    optimizer = torch.optim.AdamW(list(policy.parameters()), lr=lr)
    config = dict(trainer='llm-sft', steps=steps, lr=lr, batch_size=batch_size,
                  data=digest([asdict(t) for t in traces]), validation=digest([asdict(t) for t in validation]))
    if evaluate_at != 'step':
        config['evaluate_at'] = evaluate_at
    if groups is not None:
        config['groups'] = digest(groups)
    state = TrainingState(policy.model, optimizer, seed=seed, directory=checkpoint_dir, resume=resume, config=config)
    policy.training_state = state
    scaler = state.scaler

    def evaluation(step):
        if evaluate_at == 'step' or step in (0, steps):
            return dict(train=evaluate(policy, traces), validation=evaluate(policy, validation) if validation else None)
        return dict(train=None, validation=None)
    if not state.history:
        state.history = [dict(step=0, **evaluation(0))]
        state.save()
    for step in range(state.step, min(steps, stop_after) if stop_after is not None else steps):
        chosen = state.batch(len(traces) if groups is None else len(groups), batch_size)
        batch = ([[(traces[i], 1.)] for i in chosen] if groups is None else
                 [[(traces[i], w) for i, w in groups[g]] for g in chosen])
        policy.model.train()
        def backward(active_scaler):
            total = 0.
            for members in batch:
                for trace, weight in members:
                    with autocast_for(policy.model):
                        loss = restricted_loss(policy, [trace]) / len(batch)
                        if weight != 1.:
                            loss = loss * weight
                    active_scaler.scale(loss).backward(); total += float(loss.detach())
            return total
        total, norm, retries = scaled_step(optimizer, scaler, policy.parameters(), backward)
        state.commit(dict(optimization_loss=total, grad_norm=norm, amp_retries=retries, **evaluation(step+1)))
    return state.history


def freeze_reference(policy):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(policy.model, adapter_name='default').items()}
    set_peft_model_state_dict(policy.model, state, adapter_name='reference')
    policy.model.set_adapter('default')
    for name, parameter in policy.model.named_parameters():
        if '.reference.' in name:
            parameter.requires_grad_(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='models/qwen3-1.7b')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--validation-rows', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--prompt', choices=['compact', 'full'], default='compact')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--precision', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--evaluate-at', choices=EVALUATE_AT, default='step',
                        help='full train/validation evaluation after every step (default) or only at step 0 and the end')
    args = parser.parse_args(argv)
    from scripts.vcp_make_sft import output_path, write_json
    out = output_path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    policy = LoRAPolicy.load(args.model, device=args.device, dtype=torch.float32, precision=args.precision)
    rows = read_rows(args.data)
    traces = traces_for(policy, rows[:args.limit], args.registry, max_length=args.max_length, prompt=args.prompt)
    validation = traces_for(policy, rows[args.limit:args.limit+args.validation_rows], args.registry, max_length=args.max_length, prompt=args.prompt)
    history = train(policy, traces, validation=validation, steps=args.steps,
                    checkpoint_dir=out/'resume', resume=args.resume, precision=args.precision,
                    evaluate_at=args.evaluate_at)
    policy.save_adapter(out / 'adapter')
    write_json(out / 'metrics.json', dict(history=history, model=args.model, provenance=policy.training_state.guard,
                                         dtype=policy.training_state.guard['precision'],
                                         tokens=[len(t.token_ids) for t in traces]))


if __name__ == '__main__':
    main()
