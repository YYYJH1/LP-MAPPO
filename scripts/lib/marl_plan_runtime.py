import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from noref_common import sha

INTERFACE = {'R': 'dp_window_repair+f17', 'B': 'dp_window+f17', 'L': 'f17'}
METRICS = ('C', 'K', 'c_min', 'feasible', 'on_time', 'loss', 'energy_J', 'eta_J', 'upsilon')
COMPARE = ('energy_J', 'C', 'K', 'loss', 'feasible', 'action_fingerprint')
_CONTEXT = None
_NETWORKS = {}
_EXECUTOR = None
_PAYLOAD = None


def runtime():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(name) != '8':
            raise ValueError(name + '=8 required')
    import torch
    torch.set_num_threads(8)


def code_version():
    from lawn_mec.llm.vcp_grammar import _code_version
    return _code_version()


def context(row):
    global _CONTEXT
    if _CONTEXT is not None and _CONTEXT.sha256 == row['sha256']:
        return _CONTEXT
    import noref_common as c
    from lawn_mec.llm import vcp_grammar as vg
    from lawn_mec.vcp.grammar import Grammar
    if not os.environ.get('LAWN_CONTEXT_CACHE'):
        raise ValueError('LAWN_CONTEXT_CACHE must name the instance context store')
    path = Path(row['path']).resolve()
    if sha(path) != row['sha256']:
        raise ValueError('instance content differs from manifest')
    ctx = c.read_cached(vg._identity(path), 'nominal')
    if ctx.path != str(path) or ctx.sha256 != row['sha256'] or ctx.reference_rate != 'nominal' or ctx.tables.reference_rate != 'nominal':
        raise ValueError('cached context identity/reference rate differs')
    _CONTEXT = SimpleNamespace(tables=ctx.tables, dp=ctx.dp,
                               grammar=Grammar(ctx.tables), instance=ctx.instance, sha256=ctx.sha256)
    return _CONTEXT


def load_runs(seeds):
    import noref_common as c
    import s2_marl_eval as S
    from lawn_mec.marl_v2.train import source_digest
    loaded, proofs = {}, {}
    for seed in seeds:
        run = c.RUNS / f'S2_mappo_s{seed}'
        ck = run / 'checkpoints/latest.pt'
        h = sha(ck)
        meta, iteration, common = S.trained(run)
        if sha(ck) != h or meta['seed'] != seed or meta['algorithm'] != 'mappo' or meta['execution_layer'] != 'audit':
            raise ValueError('requires complete default same-seed MAPPO audit checkpoint')
        if not meta['flags']['f15'] or not meta['flags']['f16'] or common.get('skip_terminal_replay'):
            raise ValueError('default f15/f16 + terminal replay required')
        loaded[seed] = common
        proofs[str(seed)] = dict(run=str(run), checkpoint_sha256=h, iteration=iteration,
                                training_sources=meta['sources'], evaluated_sources=source_digest(),
                                source_mismatch=meta['sources'] != source_digest(),
                                run_sha256=sha(run/'run.json'), complete_sha256=sha(run/'complete.json'))
    return loaded, proofs


def init(payload):
    global _PAYLOAD, _EXECUTOR, _CONTEXT, _NETWORKS
    runtime()
    from lawn_mec.marl_v2.rollout import ExecutorCache
    _PAYLOAD = payload
    _EXECUTOR = ExecutorCache(capacity=1)
    _CONTEXT = None
    _NETWORKS = {}


def worker(task):
    import torch
    from lawn_mec.marl_v2.flags import Flags
    from lawn_mec.marl_v2.nets import ActorBank, Critic, ValueNorm
    from lawn_mec.marl_v2.rollout import collect
    from lawn_mec.vcpm.interface import interface_options
    from lawn_mec.vcpm.library import unpack
    common = _PAYLOAD[task['mappo_seed']]
    seed = task['mappo_seed']
    if seed not in _NETWORKS:
        flags = Flags(**dict(common['flags'], f17=True))
        bank = ActorBank(common['U'], common['shared'], flags=flags)
        critic = Critic(common['state_dim'], common['U'], common['local'], flags=flags)
        norm = ValueNorm()
        if flags.f14c:
            from lawn_mec.marl_v2.f14_value import PotentialResidualCritic
            critic = PotentialResidualCritic(critic, norm)
        for name, net in (('actors', bank), ('critic', critic), ('normalizer', norm)):
            net.load_state_dict(common[name])
        _NETWORKS[seed] = (bank, critic, norm, flags)
    bank, critic, norm, flags = _NETWORKS[seed]
    ctx = context(task['row'])
    if common['U'] != ctx.instance['params']['U']:
        raise ValueError('checkpoint/instance U differs')
    options = interface_options(INTERFACE[task['interface']])
    if options['pace'] != 'off' and task['plan'] is None:
        raise ValueError('window execution requires a validated explicit plan')
    wrapped = (SimpleNamespace(tables=ctx.tables, reference=unpack(task['plan']))
               if options['pace'] != 'off' else None)
    executor = _EXECUTOR.fresh(ctx.instance, execution_layer=common['execution_layer'], terminal_replay=True)
    tic = time.perf_counter(); cpu = time.process_time()
    extra = dict(pace_context=wrapped, pace=options['pace']) if options['pace'] != 'off' else {}
    episode = collect(ctx.instance, bank, critic, norm, seed=task['action_seed'], executor=executor, flags=flags, **extra)
    result = episode.result
    m = {k: result['metrics'][k] for k in METRICS}
    m['action_fingerprint'] = hashlib.sha256(json.dumps(result['decisions'], sort_keys=True, default=str).encode()).hexdigest()
    for name in ('pace', 'guards'):
        if name in result:
            m[name] = result[name]
    m['wall_s'] = time.perf_counter()-tic; m['cpu_s'] = time.process_time()-cpu
    if torch.cuda.is_initialized():
        raise AssertionError('CPU-only evaluation initialized CUDA')
    return m
