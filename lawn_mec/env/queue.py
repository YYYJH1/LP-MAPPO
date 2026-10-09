from dataclasses import dataclass, field
import numpy as np

@dataclass
class Event:
    available: bool = False
    value: int | None = None

    def set(self, n):
        if not self.available:
            self.available, self.value = True, int(n)

    def as_dict(self):
        return {'available': self.available, 'value': self.value}

@dataclass
class TaskState:
    location: int | None = None
    W: np.ndarray = field(default_factory=lambda: np.zeros(3))
    J: dict = field(default_factory=lambda: {'s': 0, 'a': 0})
    events: dict = field(default_factory=lambda: {x: Event() for x in ('s', 'g', 'ul', 'c', 'dl', 'r', 'a')})
    injected: set = field(default_factory=set)


def acquisition(start, length):
    return start+length


def arrivals(task, state, n):
    A = np.zeros(3)
    for event, component, workload in [('g', 0 if state.location else 1, task['D'] if state.location else task['C']),
                                       ('ul', 1, task['C']), ('c', 2, task['O'])]:
        e = state.events[event]
        if e.available and e.value == n and event not in state.injected:
            if event == 'g' and state.location is None:
                raise ValueError('execution location must be committed before release')
            if event == 'g' or state.location:
                A[component] += workload
            state.injected.add(event)
    return A


def inject(task, state, n):
    A = arrivals(task, state, n)
    state.W = state.W+A
    return A


def serve(state, mu, n):
    mu = np.asarray(mu)
    for i, stage in enumerate(('ul', 'c', 'dl')):
        if (state.location != 0 or i == 1) and 0 < state.W[i] <= mu[i]:
            state.events[stage].set(n+1)
            if stage == ('c' if state.location == 0 else 'dl'):
                state.events['r'].set(n+1)
    state.W = np.maximum(state.W-mu, 0)
    return state.W.copy()


def advance_visit(state, phase, length, n):
    if state.J[phase] >= length:
        raise ValueError('completed visit cannot run again')
    state.events[phase].set(n)
    state.J[phase] += 1
    if phase == 's' and state.J[phase] == length:
        state.events['g'].set(acquisition(state.events['s'].value, length))
