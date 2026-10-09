from lawn_mec.exec_v2.events import EventExecutor
from .rollout import collect as _collect


def collect(instance, actors, critic, normalizer, *, seed=200, commitment=None,
            executor=None, flags=None, skip_terminal_replay=False):
    if not isinstance(skip_terminal_replay, bool):
        raise ValueError('skip_terminal_replay must be a boolean')
    executor = executor or EventExecutor(instance, execution_layer='audit',
                                         terminal_replay=not skip_terminal_replay)
    if executor.state.layer != 'audit' or executor.service is None:
        raise ValueError('audit collection requires audit execution and certified service')
    if executor.terminal_replay == skip_terminal_replay:
        raise ValueError('provided executor does not match the requested terminal replay option')
    episode = _collect(instance, actors, critic, normalizer, seed=seed,
                       commitment=commitment, executor=executor, flags=flags)
    if skip_terminal_replay:
        episode.result['measurement']['includes'] = 'environment init, actor, critic, all N audit-layer steps and terminal accumulator; no terminal replay'
    return episode
