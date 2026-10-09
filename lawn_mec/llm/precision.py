from contextlib import nullcontext
import torch

CHOICES = ('float16', 'bfloat16', 'float32')

LOGPROB_AGREEMENT_BOUND = 0.02


def normalize(value):
    value = str(value).removeprefix('torch.')
    value = {'fp16':'float16', 'bf16':'bfloat16', 'fp32':'float32'}.get(value, value)
    if value not in CHOICES:
        raise ValueError(f'precision must be one of {CHOICES}')
    return value


def configure(model, precision='float16'):
    requested = normalize(precision)
    device = next(model.parameters()).device.type
    if device == 'cpu':
        torch.use_deterministic_algorithms(True)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('training requires float32 master weights')
    model._planner_precision = dict(requested=requested, compute=requested if device == 'cuda' else 'float32',
                                    master='float32', device=device,
                                    deterministic=torch.are_deterministic_algorithms_enabled())
    return model._planner_precision


def contract(model):
    if not hasattr(model, '_planner_precision'):
        configure(model)
    return model._planner_precision


def autocast(model):
    info = contract(model)
    if info['device'] == 'cuda' and info['compute'] != 'float32':
        return torch.autocast('cuda', dtype=getattr(torch, info['compute']))
    return nullcontext()


def scaler_for(model):
    info = contract(model)
    return torch.amp.GradScaler('cuda', enabled=info['device']=='cuda' and info['compute']=='float16')


def check_agreement(max_abs, *, what):
    if not max_abs <= LOGPROB_AGREEMENT_BOUND:
        raise ValueError(f'{what}: sampled and recomputed field log-probabilities differ by {max_abs:.3g} '
                         f'> {LOGPROB_AGREEMENT_BOUND}, beyond fp16 rounding; the sampler and the trainer '
                         'do not describe the same policy')
    return max_abs


def check_pair(model, sampler, *, allow_mixed_precision=False):
    trainer = contract(model)['compute']
    sampling = normalize(getattr(sampler, 'precision', trainer))
    mixed = sampling != trainer
    if mixed and not allow_mixed_precision:
        raise ValueError(f'sampler/trainer precision differs: {sampling}/{trainer}; '
                         'explicit allow_mixed_precision required')
    return dict(trainer=contract(model), sampler=sampling, allow_mixed_precision=bool(allow_mixed_precision), mixed=mixed)
