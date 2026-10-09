from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from lawn_mec.env.model import slot_capacity
from lawn_mec.env.queue import inject
from .state import clone_task


@dataclass
class Allocation:
    sigma: np.ndarray
    power: np.ndarray
    f: np.ndarray
    weights: np.ndarray
    options: dict
    backlog: np.ndarray


def boundary_tasks(state, locations, *, service=None):
    if len(locations) != len(state.tasks):
        raise ValueError('one location per task required')
    pending = [clone_task(st) for st in state.states]
    for k, st in enumerate(pending):
        m = locations[k]
        if st.events['g'].value == state.n:
            if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or not 0 <= m <= state.p['M']:
                raise ValueError(f'task {k} requires a legal explicit release location')
        elif m != st.location:
            raise ValueError(f'task {k} cannot change location')
        st.location = m
        if m is not None:
            inject(state.tasks[k], st, state.n)
        if service is not None:
            pending[k] = service.task_view(k, state.tasks[k], st, state.n)
    return pending


def maximum_weight_matching(weights):
    w = np.asarray(weights, float)
    if w.ndim != 2 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError('finite nonnegative matrix required')
    U, M = w.shape
    rows, cols = linear_sum_assignment(np.concatenate((w, np.zeros((U, U))), axis=1), maximize=True)
    return [(int(u), int(m)) for u, m in zip(rows, cols) if m < M and w[u, m] > 0]


def fractional_cpu(indices, backlog, priorities, cycles, capacity, delta):
    result = np.zeros(len(backlog))
    indices = list(indices)
    if capacity < 0 or delta <= 0:
        raise ValueError('nonnegative CPU capacity and positive delta required')
    remaining = float(capacity)
    order = sorted((k for k in indices if priorities[k] > 0 and backlog[k] > 0),
                   key=lambda k: (-priorities[k]/cycles[k], k))
    for k in order:
        result[k] = min(remaining, backlog[k]/delta)
        remaining -= result[k]
        if remaining <= 0:
            break
    allocated = [k for k in order if result[k] > 0]
    def represented_sum():
        return max(float(sum(result)), float(np.sum(result)),
                   float(np.sum(result[indices])))
    while allocated and represented_sum() > capacity:
        k = allocated[-1]
        excess = represented_sum()-capacity
        result[k] = max(0., np.nextafter(result[k]-excess, 0.))
        if result[k] == 0:
            allocated.pop()
    return result


def allocate(state, accel, locations, priorities, *, service=None):
    p, tasks = state.p, state.tasks
    K, U, M = len(tasks), p['U'], p['M']
    w, a = np.asarray(priorities, float), np.asarray(accel, float)
    if w.shape != (K, 3) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError('explicit finite nonnegative K x 3 priorities required')
    if a.shape != (U, 3) or not np.all(np.isfinite(a)):
        raise ValueError('finite U x 3 planned accelerations required')
    pending = boundary_tasks(state, locations, service=service)
    backlog = np.array([st.W for st in pending])
    weights, options, capacities = np.zeros((U, M)), {}, {}
    for k, task in enumerate(tasks):
        m, u = locations[k], task['u']
        if m in (None, 0):
            continue
        for stage, direction, key in ((0, 0, 'D'), (2, 1, 'O')):
            if w[k, stage] == 0 or backlog[k, stage] <= 0:
                continue
            cache_key = (u, m, direction)
            if cache_key not in capacities:
                power = p['P_ul'] if direction == 0 else p['P_dl']
                if state.layer == 'audit':
                    capacities[cache_key] = max(0., state.physics.radio(
                        state.q[u], state.v[u], a[u], m-1, power, 1.).lo)
                else:
                    capacities[cache_key] = slot_capacity(state.q[u], state.v[u], a[u],
                        state.instance['bs'][m-1], state.instance['buildings'], power, 1., p, 'guard', state.cfg)
            value = w[k, stage]*min(capacities[cache_key], backlog[k, stage])/task[key]
            edge = (u, m-1)
            option = (k, direction)
            if value > weights[edge] or (value == weights[edge] and value > 0 and option < options[edge]):
                weights[edge], options[edge] = value, option
    sigma, power = np.zeros((K, 2)), np.zeros(K)
    for edge in maximum_weight_matching(weights):
        k, direction = options[edge]
        sigma[k, direction] = 1.
        if direction == 0:
            power[k] = p['P_ul']
    cycles = np.array([t['C'] for t in tasks])
    f = np.zeros(K)
    for u in range(U):
        indices = [k for k, t in enumerate(tasks) if t['u'] == u and locations[k] == 0]
        f += fractional_cpu(indices, backlog[:, 1], w[:, 1], cycles, p['F_local'], p['delta'])
    for m in range(1, M+1):
        indices = [k for k in range(K) if locations[k] == m]
        f += fractional_cpu(indices, backlog[:, 1], w[:, 1], cycles,
                            p['G_total']-state.instance['load'][m-1][state.n], p['delta'])
    return Allocation(sigma, power, f, weights, options, backlog)
