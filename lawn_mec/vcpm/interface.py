from dataclasses import replace
from .env import pace_support, check_pace
from lawn_mec.marl_v2.env import MarlEnv, choice_features

INTERFACES = {'none': ('off', False), 'f17': ('off', True),
              'pace_band+f17': ('band', True), 'pace_ready+f17': ('ready', True),
              'floor_only+f17': ('floor_only', True), 's_only+f17': ('s_only', True),
              'a_only+f17': ('a_only', True), 'dp_window+f17': ('band', True),
              'pace_repair+f17': ('repair', True), 'dp_window_repair+f17': ('repair', True)}


def interface_options(name):
    pace, f17 = INTERFACES[name]
    return dict(pace=pace, f17=f17)


class DPWindowEnv(MarlEnv):
    def __init__(self, instance, *, pace_context, pace='band', **kwargs):
        super().__init__(instance, **kwargs)
        self.tables = pace_context.tables
        self.plan = replace(pace_context.reference,
                            tasks=tuple(replace(c, g='S') for c in pace_context.reference.tasks))
        self.pace = check_pace(pace)
        self.pace_counts = dict(forced=0, bind=0, empty=0, override=0)
        if self.pace == 'repair':
            if not self.flags.f15:
                raise ValueError('pace requires f15')
            from .ready_repair import init_counts
            init_counts(self)

    def _choose(self, actor, u, head, features, mask, records, request=None):
        import numpy as np
        import torch
        support, forced, empty, override = pace_support(self, head, choice_features(features), mask, request, self.pace)
        for key, value in zip(('forced', 'empty', 'override'), (forced, empty, override)):
            self.pace_counts[key] += int(value)
        if forced and not np.array_equal(support, mask):
            with torch.random.fork_rng(devices=[]):
                raw, _ = actor.sample(u, head, self.local_observation(u, request), choice_features(features),
                                      np.asarray(mask, bool))
            self.pace_counts['bind'] += int(not support[raw])
        action = super()._choose(actor, u, head, features, support, records, request)
        if self.pace == 'repair':
            from .ready_repair import record_wait
            record_wait(self, head, request, choice_features(features), action)
        return action

    def pace_stats(self):
        return dict(self.pace_counts)
