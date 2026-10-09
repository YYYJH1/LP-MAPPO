import copy
import numpy as np
from .params import checker_config
from .geometry import Box
from .queue import TaskState
from .generator import accept
from . import transaction
from .checker import replay_episode
from lawn_mec.eval.registry import aggregate, FIELDS


class Episode:
    def __init__(self, instance=None, seed=0, cfg=None):
        self.cfg = cfg or checker_config()
        if instance is not None: self.reset(instance, seed)

    def reset(self, instance, seed=0):
        result = accept(instance)
        if not result['passed']: raise ValueError(f'invalid geometry: {result["reasons"]}')
        self.instance = copy.deepcopy(instance); self.instance['_geometry_accepted'] = True
        self.p = self.instance['params']; self.tasks = self.instance['tasks']
        self.N = round(self.p['T']/self.p['delta']); self.n = 0
        self.q, self.v = np.array(instance['qI']), np.array(instance['v0'])
        self.B = np.full(self.p['U'], self.p['battery']); self.states = [TaskState() for _ in self.tasks]
        self.node = [next(i for i, b in enumerate(instance['nodes']) if Box.from_dict(b).contains(q)) for q in self.q]
        self.at_node = [True]*self.p['U']; self.edge = [None]*self.p['U']; self.edge_start = [0]*self.p['U']
        self.route = [[] for _ in range(self.p['U'])]; self.visit_pointer = [0]*self.p['U']; self.logs = []
        self.rng = np.random.default_rng(seed)
        return self.observation()

    def observation(self):
        return {'n': self.n, 'q': self.q.copy(), 'v': self.v.copy(), 'B': self.B.copy(), 'edge': list(self.edge),
                'tasks': [{'W': s.W.copy(), 'J': dict(s.J), 'location': s.location,
                           'events': {k: e.as_dict() for k, e in s.events.items()}} for s in self.states],
                'G_remaining': self.p['G_total']-np.array(self.instance['load'])[:, self.n:]}

    def masks(self, u, projected_accel=None, location=None):
        node, event = self.node[u], self.at_node[u]
        if self.edge[u] is not None:
            edge = self.instance['edges'][self.edge[u]]
            if self.n > self.edge_start[u] and Box.from_dict(self.instance['nodes'][edge['target']]).contains(self.q[u], self.cfg['epsilon_num']):
                node, event = edge['target'], True
        edges = [i for i, e in enumerate(self.instance['edges']) if e['source'] == node] if event else []
        release = any(t['u'] == u and s.events['g'].value == self.n and s.location is None for t, s in zip(self.tasks, self.states))
        return {'edge_choice': edges, 'visit_start': False if projected_accel is None else transaction.visit_allowed(self, u, np.asarray(projected_accel)),
                'location': list(range(self.p['M']+1)) if release else [],
                'request': transaction.request_options(self, u, location)}

    def step(self, joint_action):
        return transaction.step(self, joint_action)

    def tables(self, metadata=None):
        if self.n != self.N: raise RuntimeError('tables require a completed episode')
        metadata = metadata or {}; evaluation = replay_episode(self.instance, self.logs, 'evaluation', self.cfg)
        audit = replay_episode(self.instance, self.logs, 'audit', self.cfg)
        er = max(r['endpoint_residual'] for r in evaluation['uavs'])
        evstat = aggregate(evaluation['conditions'], er, self.cfg['epsilon_phys'])
        auditstat = aggregate(audit['conditions'], er, self.cfg['epsilon_phys'])
        disagreement = sum(slot['guard']['conditions'][k]['status'] != ev['conditions'][k]['status']
                           for slot, ev in zip(self.logs, evaluation['slots']) for k in FIELDS)
        disagreement += sum(ev['conditions'][k]['status'] != au['conditions'][k]['status']
                            for ev, au in zip(evaluation['slots'], audit['slots']) for k in FIELDS)
        p = self.p; tasks = evaluation['tasks']; K = len(tasks)
        total = {f'E_{name}': sum(u[f'E_{name}'] for u in evaluation['uavs']) for name in 'fcrp'}
        mec_work = sum(sum(f for f, m in zip(s['f'], s['locations']) if m not in (None, 0))*p['delta'] for s in self.logs)
        local_work = sum(sum(f for f, m in zip(s['f'], s['locations']) if m == 0)*p['delta'] for s in self.logs)
        completed = sum(t['completed'] for t in tasks)
        fields = {name: item['status'] == 'fail' for name, item in evaluation['field_conditions'].items()}
        row = {**metadata, **total, 'E_total': sum(total.values()), **evstat, 'certified_status': auditstat['feasible_status'],
               'conditions': evaluation['conditions'], 'audit_conditions': audit['conditions'],
               'field_conditions': evaluation['field_conditions'], 'audit_field_conditions': audit['field_conditions'], **fields,
               **evaluation['diagnostics'], 'path_candidate_count': None,
               'viol_projection_empty': sum(s['diagnostics']['viol_projection_empty'] for s in self.logs),
               'viol_endpoint_unreachable': sum(s['diagnostics']['viol_endpoint_unreachable'] for s in self.logs),
               'checker_unknown_count': auditstat['checker_unknown_count'], 'checker_disagreement': disagreement,
               'n_tasks': K, 'n_completed': completed,
               'offload_ratio': sum(t['location'] is not None and t['location'] > 0 for t in tasks)/K,
               'mec_util': mec_work/(p['delta']*np.sum(p['G_total']-np.array(self.instance['load']))),
               'bw_util': sum(np.sum(s['sigma']) for s in self.logs)/(p['M']*self.N),
               'cpu_local_util': local_work/(p['U']*p['T']*p['F_local']),
               'inference_time_ms': metadata.get('inference_time_ms'),
               'wall_time_s': sum(s['wall_time_s'] for s in self.logs),
               'guard_mean_slot_s': np.mean([s['guard_wall_time_s'] for s in self.logs]),
               'evaluation_wall_time_s': evaluation['wall_time_s'], 'audit_wall_time_s': audit['wall_time_s']}
        return {'episode': row, 'uav': [{**metadata, **r} for r in evaluation['uavs']],
                'task': [{**metadata, **r} for r in tasks], 'train': [],
                'diag': [{**metadata, 'group': 'episode', **evaluation['diagnostics'], 'checker_disagreement': disagreement}],
                'evaluation': evaluation, 'audit': audit}
