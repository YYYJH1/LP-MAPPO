from dataclasses import dataclass
import weakref

from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.utils.weak import WeakIdKeyDictionary

_owners = WeakIdKeyDictionary()
_versions = WeakIdKeyDictionary()
_hook = None


@dataclass(frozen=True)
class PolicyIdentity:
    version: int
    writes: tuple[tuple[int, int], ...]


def _count_step(optimizer, args, kwargs):
    stepped = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for module in _owners.get(parameter, ()):
                stepped[id(module)] = module
    for module in stepped.values():
        _versions[module] += 1


def track(module):
    global _hook
    if _hook is None:
        _hook = register_optimizer_step_post_hook(_count_step)
    _versions.setdefault(module, 0)
    for parameter in module.parameters():
        owners = _owners.get(parameter)
        if owners is None:
            owners = _owners[parameter] = weakref.WeakSet()
        owners.add(module)
    return module


def policy_version(module) -> int:
    track(module)
    return _versions[module]


def set_policy_version(module, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('policy version must be a nonnegative integer')
    track(module)
    _versions[module] = value


def policy_identity(module) -> PolicyIdentity:
    version = policy_version(module)
    return PolicyIdentity(version, tuple((id(p), p._version) for p in module.parameters()))


def require_fresh(module, sampled, *, what='tree', remedy='resample for an on-policy update'):
    if sampled is None:
        raise ValueError(f'{what} lacks its sampling policy identity; {remedy}')
    current = policy_identity(module)
    if current.version != sampled.version:
        raise ValueError(f'{what} is stale: policy version {sampled.version} at sampling, '
                         f'{current.version} now (optimizer steps in between); {remedy}')
    if current.writes != sampled.writes:
        raise ValueError(f'{what} is stale: parameters were written in place after sampling; {remedy}')
    return current
