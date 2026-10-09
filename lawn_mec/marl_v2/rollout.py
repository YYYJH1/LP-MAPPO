from time import perf_counter
from collections import OrderedDict
import hashlib
import inspect
import json
import os
import numpy as np
import torch
from .env import MarlEnv
from .batch import Episode
from lawn_mec.exec_v2.events import EventExecutor


class ExecutorCache:
    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError('executor cache capacity must be positive')
        self.capacity = capacity
        self._templates = OrderedDict()

    def fresh(self, instance, *, execution_layer='evaluation', terminal_replay=True):
        if execution_layer not in ('evaluation', 'audit'):
            raise ValueError('execution layer must be evaluation or audit')
        if not isinstance(terminal_replay, bool):
            raise ValueError('terminal_replay must be a boolean')
        if not terminal_replay and execution_layer != 'audit':
            raise ValueError('skipping terminal replay requires audit execution')
        key = (execution_layer, terminal_replay, hashlib.sha256(json.dumps(instance,sort_keys=True,separators=(',',':'),
                                                        allow_nan=False).encode()).digest())
        if key not in self._templates:
            if 'execution_layer' in inspect.signature(EventExecutor).parameters:
                options = dict(execution_layer=execution_layer)
            elif execution_layer == 'evaluation':
                options = {}
            else:
                raise ValueError('this executor does not expose audit execution')
            if 'terminal_replay' in inspect.signature(EventExecutor).parameters:
                options['terminal_replay'] = terminal_replay
            elif not terminal_replay:
                raise ValueError('this executor does not expose terminal replay selection')
            self._templates[key] = EventExecutor(instance, **options)
            if len(self._templates) > self.capacity:
                self._templates.popitem(last=False)
        self._templates.move_to_end(key)
        return self._templates[key].clone()


def collect(instance,actors,critic,normalizer,*,seed=200,commitment=None,executor=None,flags=None,
            pace_context=None, pace='off'):
    if next(actors.parameters()).device.type!='cpu' or next(critic.parameters()).device.type!='cpu':
        raise ValueError('rollouts must run on CPU')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    actors.eval(); critic.eval()
    tic=perf_counter()
    flags = getattr(actors, 'flags', None) if flags is None else flags
    if pace != 'off':
        from lawn_mec.vcpm.interface import DPWindowEnv
        env=DPWindowEnv(instance,executor=executor,flags=flags,pace_context=pace_context,pace=pace)
    else:
        env=MarlEnv(instance,commitment=commitment,executor=executor,flags=flags)
    records=[]; states=[]; local=[]; rewards=[]
    for _ in range(env.N):
        full=env.critic_state()
        obs=[env.critic_local_observation(u) for u in range(env.U)]
        states.append(full); local.append(obs)
        actions,reward,done=env.step(actors)
        records.extend(actions); rewards.append(reward)
    if not done or env.executor.state.n!=env.N:
        raise AssertionError('incomplete rollout')
    result=env.last_result
    if env.flags.f17:
        result['guards'] = env.guard_stats()
    if pace != 'off':
        result['pace'] = env.pace_stats()
    error=abs(sum(rewards)+result['metrics']['loss'])
    if error>1e-8:
        raise AssertionError(f'reward telescoping error {error}')
    with torch.inference_mode():
        if critic.local:
            pred=torch.stack([critic(obs) for obs in local])
        else:
            pred=critic(states)[:,None].expand(-1,env.U)
        normalized=pred.numpy().copy()
        values=normalizer.denormalize(pred).numpy().copy()
    wall=perf_counter()-tic
    result['measurement']=dict(t_ep_s=wall,includes='environment init, actor, critic, all N steps and frozen audit',
                               reward_identity_error=error,nice=os.getpriority(os.PRIO_PROCESS,0),
                               torch_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized(),
                               executor_reused=executor is not None,affinity=sorted(os.sched_getaffinity(0)))
    return Episode(records,states,local,np.asarray(rewards,np.float32),np.asarray(values),
                   normalized,result,wall)
