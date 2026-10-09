from collections import OrderedDict
import pickle
from types import FunctionType
import numpy as np

from lawn_mec.env import checker
from lawn_mec.env.interval import iv, enclose_integral
from lawn_mec.env.model import (array_signature, rate_interval, shaft_interval,
    slot_capacity, flight_energy, frequency, cpu_energy, radio_energy, payload_energy)
from .weakmemo import weak_method_lru_cache


def _array(key):
    return np.frombuffer(key[2], dtype=key[0]).reshape(key[1])


class Physics:
    def __init__(self, instance, cfg):
        self.instance, self.cfg = instance, cfg
        self.p, self.tasks = instance['params'], instance['tasks']

    @weak_method_lru_cache(maxsize=32768)
    def _radio(self, q, v, a, station, power, sigma, functions):
        power, sigma = pickle.loads(power), pickle.loads(sigma)
        args = [_array(q), _array(v), _array(a), self.instance['bs'][station],
                self.instance['buildings'], power, sigma, self.p]
        return enclose_integral(lambda t: rate_interval(*args, t), self.p['delta'],
                                self.cfg['integral_subdivisions'])

    @weak_method_lru_cache(maxsize=128)
    def _idle(self, functions):
        return enclose_integral(lambda t: iv(0), self.p['delta'], self.cfg['integral_subdivisions'])

    def radio(self, q, v, a, station, power, sigma):
        functions = (enclose_integral, rate_interval)
        if not sigma or power == 0:
            return self._idle(functions)
        return self._radio(*(array_signature(x) for x in (q, v, a)), station,
                           pickle.dumps(power), pickle.dumps(sigma), functions)

    @weak_method_lru_cache(maxsize=8192)
    def _flight(self, v, a, functions):
        v, a, p = _array(v), _array(a), self.p
        return p['delta']*p['P0']+enclose_integral(
            lambda t: shaft_interval(v, a, p, t), p['delta'],
            self.cfg['integral_subdivisions'])/p['eta_flight']

    def integrals(self, slot, layer):
        p, tasks, cfg = self.p, self.tasks, self.cfg
        dt = p['delta']
        mu = [[iv(0), iv(dt)*x, iv(0)] for x in slot['f']] if layer == 'audit' else np.zeros((len(tasks), 3))
        if layer != 'audit':
            mu[:, 1] = dt*np.array(slot['f'])
        q, velocities, accel = (np.array(slot[key]) for key in ('q', 'v', 'accel'))
        energy = []
        for k, t in enumerate(tasks):
            m, u = slot['locations'][k], t['u']
            if m in (0, None):
                continue
            for j, component in [(0, 0), (1, 2)]:
                sigma = slot['sigma'][k][j]
                power = slot['power'][k] if j == 0 else p['P_dl']*sigma
                if layer != 'audit' and (not sigma or power == 0):
                    continue
                if layer == 'audit':
                    value = self.radio(q[u], velocities[u], accel[u], m-1, power, sigma)
                else:
                    value = slot_capacity(q[u], velocities[u], accel[u], self.instance['bs'][m-1],
                                          self.instance['buildings'], power, sigma, p, layer, cfg)
                mu[k][component] = value
        for u in range(p['U']):
            ks = [k for k, t in enumerate(tasks) if t['u'] == u]
            v, a = velocities[u], accel[u]
            if layer == 'audit':
                ef = self._flight(array_signature(v), array_signature(a), (enclose_integral, shaft_interval))
            else:
                ef = flight_energy(v, a, p, layer, cfg)
            F = frequency(slot['f'], [t['u'] for t in tasks], slot['locations'], u)
            ec = cpu_energy(F, p)
            er = radio_energy([slot['power'][k] for k in ks], [slot['sigma'][k][0] for k in ks],
                              [slot['sigma'][k][1] for k in ks], p)
            ep = payload_energy([(k, phase) for k, phase in slot['active'] if k in ks], tasks, p)
            if layer == 'audit':
                F_interval = sum((iv(slot['f'][k]) for k in ks if slot['locations'][k] == 0), iv(0))
                ec = iv(dt)*p['kappa']*F_interval**3
                er = iv(dt)*(sum((iv(slot['power'][k]) for k in ks), iv(0))/p['eta_tx']
                              +iv(p['P_tx'])*sum(slot['sigma'][k][0] for k in ks)
                              +iv(p['P_rx'])*sum(slot['sigma'][k][1] for k in ks))
                ep = iv(dt)*sum((iv(tasks[k]['P_'+phase]) for k, phase in slot['active'] if k in ks), iv(0))
            energy.append([ef, ec, er, ep])
        return mu, energy

    def replay(self, instance, logs, layer='audit'):
        original = checker.replay_episode
        namespace = dict(original.__globals__)
        namespace['slot_integrals'] = lambda inst, slot, layer, cfg: self.integrals(slot, layer)
        replay = FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)
        return replay(instance, logs, layer, self.cfg)


_CONTEXTS = OrderedDict()


def context(instance, cfg):
    from .state import freeze, thaw
    inst, settings = freeze(instance), freeze(cfg)
    key = pickle.dumps((thaw(inst), thaw(settings)))
    if key not in _CONTEXTS:
        _CONTEXTS[key] = Physics(inst, settings)
        if len(_CONTEXTS) > 16:
            _CONTEXTS.popitem(last=False)
    _CONTEXTS.move_to_end(key)
    return _CONTEXTS[key]


def audit_radio_capacity(state, slot, k, stage):
    p = state.p
    m, u = slot['locations'][k], state.tasks[k]['u']
    if m in (None, 0):
        return iv(0)
    sigma = slot['sigma'][k][stage//2]
    power = slot['power'][k] if stage == 0 else p['P_dl']*sigma
    q, v, a = (np.asarray(slot[key])[u] for key in ('q', 'v', 'accel'))
    return state.physics.radio(q, v, a, m-1, power, sigma)
