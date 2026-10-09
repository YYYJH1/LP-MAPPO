import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .policy_version import policy_version, set_policy_version, track
from .precision import contract, scaler_for
from .sampler import derive_seed
from lawn_mec.vcp.payload import provenance, require_provenance


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':')).encode()).hexdigest()


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), n[2], n[3], n[4]),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(value):
    random.setstate(value['python'])
    n = value['numpy']; np.random.set_state((n[0], np.asarray(n[1],dtype=np.uint32), *n[2:]))
    torch.set_rng_state(value['cpu'])
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device inventory differs')
        torch.cuda.set_rng_state_all(value['cuda'])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)


def archive_attempt(directory, protocol):
    from datetime import datetime, timezone
    directory = Path(directory)
    checkpoints = sorted((directory/'grpo-resume').glob('step-????????.pt'))
    steps = 0
    if checkpoints:
        saved = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
        if digest(saved['guard']['config'].get('protocol')) != digest(protocol):
            raise ValueError('interrupted attempt protocol differs; refusing retry')
        try:
            require_provenance(saved['guard'])
        except ValueError:
            raise ValueError('interrupted attempt payload differs') from None
        steps = saved['step']
    archived = directory/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    archived.mkdir(parents=True, exist_ok=False)
    for item in list(directory.iterdir()):
        if item.name != 'attempts' and not item.name.startswith('orchestra-'):
            item.rename(archived/item.name)
    pending = json.loads((archived/'inflight.json').read_text()) if (archived/'inflight.json').exists() else {}
    pending_charge = int(pending.get('planned_verifier_calls', 0)) if pending.get('step', 0) > steps else 0
    sft = sorted((archived/'sft-resume').glob('step-????????.pt'))
    sft_steps = torch.load(sft[-1], map_location='cpu', weights_only=True)['step'] if sft else 0
    (archived/'restart.json').write_text(json.dumps(dict(
        rule='an interrupted run restarts from scratch', committed_lost_steps=steps,
        discarded_sft_steps=sft_steps,
        charged_verifier_calls=steps*protocol.get('verifier_calls_per_step',0)+pending_charge,
        budget='all discarded SFT/GRPO steps, attempted verifier calls and elapsed compute count in total consumed budget; '
               'the retry still owes the full planned successful training budget',
        uncommitted_work='charge the full dispatched in-flight verifier batch conservatively'), indent=2)+'\n')
    return archived


def weights(model):
    if hasattr(model, 'peft_config'):
        return {k:v for k,v in model.state_dict().items() if 'lora_' in k}
    return model.state_dict()


def identity(model):
    def canonical(value):
        if isinstance(value, dict):
            return {k:canonical(v) for k,v in value.items()}
        if isinstance(value, (set, frozenset)):
            return sorted(canonical(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [canonical(v) for v in value]
        return value
    config = getattr(model, 'config', None)
    config = config.to_dict() if hasattr(config, 'to_dict') else str(config)
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and 'lora_' not in name:
            h.update(name.encode()); h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return dict(type=f'{type(model).__module__}.{type(model).__qualname__}', config=config,
                schema=str(getattr(model, 'schema', None)), base_sha256=h.hexdigest(),
                adapters={k:json.loads(json.dumps(canonical(v.to_dict()), default=str))
                          for k,v in getattr(model, 'peft_config', {}).items()},
                trainable=[(n,list(p.shape)) for n,p in model.named_parameters() if p.requires_grad])


class TrainingState:
    def __init__(self, model, optimizer, *, config, seed, directory=None, resume=None,
                 reference=None, scheduler=None, scaler=None):
        if resume not in (None, 'auto') and not Path(resume).is_file():
            raise ValueError(f'resume checkpoint missing: {resume}')
        self.model, self.optimizer, self.reference = model, optimizer, reference
        track(model)
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
        self.scaler = scaler or scaler_for(model)
        self.directory = Path(directory) if directory else None
        if resume == 'auto' and self.directory is None:
            raise ValueError('resume auto requires checkpoint directory')
        self.guard = dict(format=1, config=config, seed=int(seed), model=identity(model),
                          precision=contract(model), **provenance(), sampler_seed_schedule='blake2b(base,step,v1)',
                          scheduler=dict(type=type(self.scheduler).__qualname__, initial=self.scheduler.state_dict()))
        self.guard_sha256 = digest(self.guard)
        self.seed, self.step, self.history = int(seed), 0, []
        self.order, self.position = [], 0
        self.generator = torch.Generator().manual_seed(seed)
        self.extra = {}
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        path = self.latest() if resume == 'auto' else Path(resume) if resume else None
        if path is not None:
            self.load(path)
        elif self.directory and self.latest() is not None:
            raise ValueError('checkpoint exists; select resume auto or a fresh output directory')

    def latest(self):
        files = sorted(self.directory.glob('step-????????.pt')) if self.directory else []
        return files[-1] if files else None

    def sampler_seed(self):
        return derive_seed(self.seed, 'step', self.step, 'v1')

    def batch(self, size, batch_size, *, reshuffle=False):
        if not self.order or reshuffle:
            self.order = torch.randperm(size, generator=self.generator).tolist(); self.position = 0
        ids = [self.order[(self.position+i) % size] for i in range(min(size,batch_size))]
        self.position = (self.position+batch_size) % size
        return ids

    def snapshot(self):
        return dict(guard=self.guard, guard_sha256=self.guard_sha256, weights=weights(self.model),
            reference=self.reference.state_dict() if self.reference is not None else None,
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(), step=self.step, history=self.history,
            policy_version=policy_version(self.model),
            data_order=dict(permutation=self.order, position=self.position, generator=self.generator.get_state()),
            rng=rng_state(), sampler_seed=dict(base=self.seed, next=self.sampler_seed()), extra=self.extra)

    def save(self):
        if self.directory is None:
            return
        target = self.directory / f'step-{self.step:08d}.pt'
        temporary = target.with_suffix('.pt.part')
        with temporary.open('wb') as stream:
            torch.save(self.snapshot(), stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        for old in sorted(self.directory.glob('step-????????.pt'))[:-2]:
            old.unlink()

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['guard_sha256'] != self.guard_sha256 or digest(state['guard']) != self.guard_sha256:
            raise ValueError('checkpoint config, payload, model or precision differs')
        expected = weights(self.model)
        if expected.keys() != state['weights'].keys():
            raise ValueError('checkpoint trainable parameter inventory differs')
        merged = self.model.state_dict(); merged.update(state['weights']); self.model.load_state_dict(merged)
        set_policy_version(self.model, int(state.get('policy_version', state['step'])))
        if self.reference is not None:
            if state['reference'] is None:
                raise ValueError('missing frozen reference')
            self.reference.load_state_dict(state['reference'])
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.step, self.history = state['step'], state['history']
        self.order, self.position = state['data_order']['permutation'], state['data_order']['position']
        self.generator.set_state(state['data_order']['generator'])
        self.extra = state['extra']
        if state['sampler_seed'] != dict(base=self.seed,next=self.sampler_seed()):
            raise ValueError('sampler seed schedule differs')
        restore_rng(state['rng'])

    def commit(self, metrics):
        self.step += 1; self.scheduler.step()
        self.history.append(dict(metrics, step=self.step, precision=self.guard['precision']))
        self.save()
