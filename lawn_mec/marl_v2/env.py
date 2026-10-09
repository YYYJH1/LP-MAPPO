from dataclasses import dataclass
import math
import numpy as np

from lawn_mec.env.geometry import Box, segment_box
from lawn_mec.exec_v2.events import EventExecutor
from lawn_mec.exec_v2.state import thaw
from lawn_mec.exec_v2.p0_loss import terminal_loss, c_min
from .flags import flags_or_default
from .audit_features import AuditFeatures, deadline_failures

TOKEN_DIM = 48
CHOICE_DIM = 16
HEADS = ('layer', 'bin', 'offset', 'route', 'location', 'priority')
ALL_HEADS = (*HEADS, 'timing')
PRIORITIES = (0., *tuple(float(x) for x in np.geomspace(1/16,16,16)))
COMMITMENT_WIDTH = 40


def token(kind, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) > TOKEN_DIM-8:
        raise ValueError('entity feature exceeds schema')
    row = np.zeros(TOKEN_DIM, np.float32)
    row[kind] = 1
    row[8:8+len(values)] = values
    return row


def choice_features(values):
    rows = np.zeros((len(values), CHOICE_DIM), np.float32)
    for i, value in enumerate(values):
        if len(value) > CHOICE_DIM:
            raise ValueError('candidate feature exceeds schema')
        rows[i, :len(value)] = value
    return rows


@dataclass
class HeadAction:
    uav: int
    slot: int
    head: str
    observation: np.ndarray
    candidates: np.ndarray
    mask: np.ndarray
    action: int
    old_logp: float


def potential(executor, weight=1.0):
    s = executor.state
    if s.n == s.N:
        return 0.0
    progress = []
    for task, st in zip(s.tasks, s.states):
        visits = [min(st.J[p]/task['L_'+p], 1.) for p in ('s', 'a')]
        if st.location is None:
            progress.append(visits[0]/3)
            continue
        stages = (1,) if st.location == 0 else (0, 1, 2)
        parts = visits.copy()
        for i in stages:
            end = ('ul', 'c', 'dl')[i]
            release = ('g', 'g' if st.location == 0 else 'ul', 'c')[i]
            if st.events[end].available:
                parts.append(1.)
            elif st.events[release].available:
                pending_injection = release not in st.injected
                parts.append(0. if pending_injection else
                             1.-float(st.W[i])/task[('D', 'C', 'O')[i]])
            else:
                parts.append(0.)
        progress.append(float(np.mean(parts)))
    return weight*float(np.mean(progress))


def terminal_metrics(result, instance, zeta=.85):
    audit = result.get('audit')
    if audit is None:
        audit = dict(tasks=[dict(completed=k < result['C']) for k in range(len(instance['tasks']))],
                     uavs=[dict(E_u=e) for e in result['energies_J']],
                     conditions=result['conditions'], field_conditions=result.get('field_conditions', {}))
    return dict(terminal_loss(audit, instance, zeta), conditions=audit['conditions'])


def _numbers(value):
    if isinstance(value, dict):
        for k in sorted(value):
            yield from _numbers(value[k])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float, bool)):
        yield float(value)


class MarlEnv:
    def __init__(self, instance, *, commitment=None, potential_weight=1., executor=None, flags=None):
        self.flags = flags_or_default(flags)
        self.instance = instance
        self.executor = executor or EventExecutor(instance)
        self.commitment = commitment
        self.potential_weight = potential_weight
        self.U = self.executor.state.p['U']
        self.N = self.executor.state.N
        self.K = len(instance['tasks'])
        self.prefix = [[] for _ in range(self.U)]
        keys = ('params', 'tasks', 'visits', 'nodes', 'edges', 'buildings',
                'bs', 'load', 'qI', 'qF', 'v0')
        vals = np.array(list(_numbers({k: instance[k] for k in keys})), np.float32)
        self.static = np.sign(vals)*np.log1p(np.abs(vals))
        self.capacity_history = np.zeros((self.N, self.K, 3, 2), np.float32)
        self.frequency_history = np.zeros((self.N, self.K), np.float32)
        self.last_result = None
        self._buildings = tuple(Box.from_dict(b) for b in instance['buildings'])
        self._centres = np.asarray([Box.from_dict(b).center for b in instance['nodes']])
        self._bs = np.asarray(instance['bs'])
        self._own_tasks = [[k for k,t in enumerate(self.executor.state.tasks) if t['u']==u]
                           for u in range(self.U)]
        self._task_features = []
        p = self.executor.state.p
        for k,t in enumerate(self.executor.state.tasks):
            row = np.zeros(29, np.float32)
            row[:8] = [k/max(1,self.K),t['D']/1e8,t['C']/1e10,t['O']/1e7,
                       t['L_s']/self.N,t['L_a']/self.N,t['d']/p['T'],t['H']/p['T']]
            self._task_features.append(row)
        loads = np.asarray(instance['load'],float)
        self._load_samples = np.asarray([
            loads[:,n+np.linspace(0,self.N-n-1,4,dtype=int)]/p['G_total']
            for n in range(self.N)])
        self._los = [None]*self.U
        self._event_observations = None
        self._plans = np.zeros((self.U,self.N,5),np.float32)
        self._plan_schedules = [None]*self.U
        self.audit_features = AuditFeatures(self.executor)
        self._failed = np.zeros(self.K, bool)
        self._initial_D = int(self.deadline_failures().sum()) if self.flags.f2 else 0
        self._observed_reward = 0.
        self._readiness = {}
        self._placement = None
        self.guard_counts = dict(location=[0, 0], stop_loss=[0, 0])

    def guard_stats(self):
        from .guards import guard_summary
        return guard_summary(self.guard_counts)

    def deadline_failures(self):
        self._failed |= deadline_failures(self.executor, self.audit_features)
        return self._failed.copy()

    def phi(self):
        if self.executor.state.n == self.N:
            return 0.
        if self.flags.f2:
            return -(int(self.deadline_failures().sum())-self._initial_D)/c_min(self.K)
        return potential(self.executor, self.potential_weight)

    def critic_state(self):
        if not self.flags.f4:
            state = self.full_state()
        else:
            from .state_schema import encode
            state = encode(self)
        return np.r_[state,np.float32(self._observed_reward)] if self.flags.f14c else state

    def critic_local_observation(self, u):
        observation = self.local_observation(u,with_prefix=False)
        if self.flags.f14c:
            observation = np.r_[observation,token(7,[self._observed_reward])[None]]
        return observation

    def task_states(self, *, critic=False):
        if self.executor.state.layer == 'audit':
            return self.executor.task_states()
        if (self.flags.f6 or critic) and hasattr(self.executor, 'task_states'):
            return self.executor.task_states()
        return self.executor.state.states

    def request_features(self, request, T=None):
        visit = request.get('visit')
        reason = request.get('reason')
        identity = [-1 if visit is None else visit[0]/24,
                    float(visit is not None and visit[1] == 's'),
                    float(visit is not None and visit[1] == 'a'), float(visit is None),
                    *[float(reason == r) for r in ('initial','visit_end','deferral')]]
        if visit is None:
            return [*identity, 0, 0, 0, 0]
        k, phase = visit; s = self.executor.state; task, st = s.tasks[k], self.task_states()[k]
        T = s.n if T is None else T
        ns = st.events['s'].value; ready = st.events['r'].value
        return [*identity, (task['d']-(T+task['L_a'])*s.p['delta'])/s.p['T'],
                0 if ns is None else (task['H']-(T-ns)*s.p['delta'])/s.p['T'],
                float(ns is not None), float(ready is not None and ready <= T)]

    def readiness(self, request):
        s = self.executor.state
        key = (s.n, request['uav'], request['pointer'], request['reason'])
        if key not in self._readiness:
            from .readiness import request_readiness
            self._readiness[key] = request_readiness(self, request)
        return self._readiness[key]

    def readiness_request(self, request):
        from .readiness import REQUEST_VALUES
        r = self.readiness(request)
        return list(r.values) if r is not None else [0.]*REQUEST_VALUES

    def readiness_candidate(self, request, T):
        r = self.readiness(request)
        return r.candidate(T, self.N) if r is not None else [0., 0.]

    def location_readiness(self, k, locations):
        from .location_readiness import EventCache, location_values
        if self._placement is None or self._placement.n != self.executor.state.n:
            self._placement = EventCache(self)
        return location_values(self, k, locations, self._placement)

    def contended(self, event):
        active = [tuple(x) for x in event['active']]; s = self.executor.state
        def resources(k, i):
            u = s.tasks[k]['u']; m = self.executor.locations[k]
            if i == 1:
                return {('local',u)} if m == 0 else {('cpu',m)}
            return {('radio_uav',u), ('radio_bs',m)}
        sets = {x: resources(*x) for x in active}
        return {x for x in active if any(x != y and sets[x] & sets[y] for y in active)}

    def _base_observation(self, u):
        s = self.executor.state; ex = self.executor; p = s.p
        task_states = self.task_states()
        n = s.n; dt = p['delta']; energy_scale = p['battery']
        layer = -1 if ex.layers is None else ex.layers[u]
        own = [u/max(1,self.U-1), n/self.N, *s.q[u]/1000., *s.v[u]/20.,
               (s.B[u].lo+s.B[u].hi)/2/energy_scale, layer/7,
               s.visit_pointer[u]/max(1,len(self.instance['visits'][u])),
               (s.edge[u] if s.edge[u] is not None else -1)/max(1,len(self.instance['edges']))]
        rows = np.zeros((self.U+len(self._own_tasks[u])+p['M'],TOKEN_DIM),np.float32)
        rows[0,0] = 1; rows[0,8:8+len(own)] = own
        cursor = 1
        for v in range(self.U):
            if v == u:
                continue
            c = ex.commitments[v]
            rows[cursor,1] = 1
            rows[cursor,8:19] = [v/max(1,self.U-1), *(s.q[v]-s.q[u])/1000.,
                *s.v[v]/20., -1 if ex.layers is None else ex.layers[v]/7,
                -1 if c is None else c['T']/self.N,
                -1 if c is None else c['route']/4,
                -1 if c is None else c['pointer']/max(1,len(self.instance['visits'][v]))]
            cursor += 1
        for k in self._own_tasks[u]:
            t, st = s.tasks[k], task_states[k]
            events = [(-1 if st.events[x].value is None else st.events[x].value/self.N)
                      for x in ('s','g','ul','c','dl','r','a')]
            fresh = 1. if st.events['s'].value is None else (t['H']-(n-st.events['s'].value)*dt)/t['H']
            vals = self._task_features[k].copy()
            vals[8:23] = [(t['d']-t['L_a']*dt-n*dt)/p['T'], fresh,
                    st.J['s']/t['L_s'], st.J['a']/t['L_a'],
                    *[st.W[i]/t[w] for i,w in enumerate(('D','C','O'))],
                    (-1 if st.location is None else st.location)/(p['M']+1), *events]
            vals[23:26] = (self._centres[t['z_s']]-s.q[u])/1000.
            vals[26:29] = (self._centres[t['z_a']]-s.q[u])/1000.
            rows[cursor,2] = 1; rows[cursor,8:37] = vals
            cursor += 1
        position = s.q[u].tobytes()
        if self._los[u] is None or self._los[u][0] != position:
            self._los[u] = (position,[not any(segment_box(s.q[u],bs,b) for b in self._buildings)
                                      for bs in self._bs])
        for m, bs in enumerate(self._bs):
            los = self._los[u][1][m]
            samples = self._load_samples[min(n,self.N-1),m]
            backlog = sum(float(st.W[1]) for st in task_states if st.location == m+1)/1e10
            rows[cursor,3] = 1
            rows[cursor,8:18] = [m/max(1,p['M']), *(bs-s.q[u])/1000.,float(los),backlog,*samples]
            cursor += 1
        return rows

    def local_observation(self, u, request=None, *, with_prefix=True):
        cache = self._event_observations
        if cache is None:
            base = self._base_observation(u)
        else:
            if u not in cache: cache[u] = self._base_observation(u)
            base = cache[u]
        rows = []
        if request:
            if self.flags.f6:
                values = self.request_features(request)
                if self.flags.f15:
                    values = [*values, *self.readiness_request(request)]
                rows.append(token(4, values))
            for r in request['routes']:
                rows.append(token(4, [r['route']/4, r['earliest']/self.N, r['latest']/self.N,
                    r['travel_slots']/self.N, (r['energy_J'] or 0)/self.executor.state.p['battery'], float(r['legal'])]))
        if self.commitment is not None:
            values = self.commitment_features(u)
            for row in np.atleast_2d(values):
                rows.append(token(6, row))
        if with_prefix:
            rows.extend(self.prefix[u])
        return np.concatenate([base,np.stack(rows)]) if rows else base.copy()

    def commitment_features(self,u):
        if self.commitment is None:
            return np.zeros(COMMITMENT_WIDTH,np.float32)
        value=np.asarray(self.commitment(self,u) if callable(self.commitment)
                         else self.commitment[u],np.float32).reshape(-1)
        if len(value)>COMMITMENT_WIDTH or not np.isfinite(value).all():
            raise ValueError('commitment hook requires at most 40 finite features per UAV')
        return np.pad(value,(0,COMMITMENT_WIDTH-len(value)))

    def full_state(self):
        s = self.executor.state; ex = self.executor
        values = [s.n/self.N, *s.q.flatten()/1000, *s.v.flatten()/20]
        for u in range(self.U):
            values.extend([s.visit_pointer[u], s.route_node[u], s.arrival[u], s.arrival_flags[u],
                           -1 if s.edge[u] is None else s.edge[u],
                           -1 if ex.layers is None else ex.layers[u],
                           int(u in ex.pending), int(u in ex.deferrals)])
            for energy in [*s.energies[u], s.B[u]]:
                values.extend([energy.lo/s.p['battery'], energy.hi/s.p['battery']])
            c = ex.commitments[u]
            values.extend([-1]*5 if c is None else [c['T']/self.N,c['route'],c['pointer'],c['endpoint'],c['end']/self.N])
            values.append(int(c is not None and (u,c['pointer'],s.n) in ex.approved_starts))
        for k,(t,st) in enumerate(zip(s.tasks,s.states)):
            values.extend([st.W[i]/t[w] for i,w in enumerate(('D','C','O'))])
            values.extend([st.J['s'],st.J['a'], -1 if st.location is None else st.location,
                           -1 if ex.locations[k] is None else ex.locations[k]])
            for x in ('s','g','ul','c','dl','r','a'):
                values.extend([st.events[x].available,-1 if st.events[x].value is None else st.events[x].value/self.N,
                               int(x in st.injected)])
        for group in (s.conditions,s.fields):
            for key in sorted(group):
                v=group[key]; r=v['residual']
                values.extend([{'pass':0.,'unknown':1.,'fail':2.}[v['status']],
                               math.copysign(math.log1p(abs(r)),r)])
        for u, schedule in enumerate(ex.schedules):
            if schedule is not self._plan_schedules[u]:
                self._plans[u].fill(0)
                for n,row in enumerate(schedule[:self.N]):
                    self._plans[u,n] = [1,row['edge'],*row['accel']]
                self._plan_schedules[u] = schedule
        return np.concatenate([self.static,np.asarray(values,np.float32),self._plans.ravel(),
                               self.capacity_history.ravel(),self.frequency_history.ravel(),
                               np.asarray([float(self.commitment is not None)],np.float32),
                               np.concatenate([self.commitment_features(u) for u in range(self.U)])])

    @staticmethod
    def _legal_times(req, lo, hi):
        mask = np.zeros(hi-lo+1,bool)
        for r in req['routes']:
            if r['legal']:
                start=max(lo,r['earliest']); end=min(hi,r['latest'])
                if start<=end: mask[start-lo:end-lo+1]=True
        if req['keep']: mask[req['keep_T']-lo]=True
        return mask

    def _apply(self, decision):
        self.executor.apply(decision)
        self._event_observations.clear()
        self._readiness.clear()
        self._placement = None

    def _choose(self, actor, u, head, features, mask, records, request=None):
        mask = np.asarray(mask,bool)
        if self.flags.f17:
            from .guards import guard_support
            mask = guard_support(self, head, features, mask, request)
        if not mask.any():
            raise ValueError(f'empty executor support for {head}')
        observation = self.local_observation(u,request)
        candidates = choice_features(features)
        action, logp = actor.sample(u,head,observation,candidates,mask)
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('actor selected a masked choice; no repair or resampling')
        records.append(HeadAction(u,self.executor.state.n,head,observation,candidates,mask,action,logp))
        self.prefix[u].append(token(5,[ALL_HEADS.index(head)/len(HEADS),*candidates[action]]))
        return action

    def step(self, actor):
        self._event_observations = {}
        try:
            return self._step(actor)
        finally:
            self._event_observations = None

    def _step(self, actor):
        ex=self.executor; n=ex.state.n
        if n == self.N:
            raise RuntimeError('finished')
        self.prefix=[[] for _ in range(self.U)]
        records=[]; before=self.phi()
        while ex.state.n == n:
            event=ex.next_event(); kind=event['kind']
            if kind == 'layers':
                layers=[]
                for u in event['uavs']:
                    mask=[legal and i not in layers for i,legal in enumerate(event['masks'][u])]
                    a=self._choose(actor,u,'layer',[[i/7,h/100] for i,h in enumerate(event['heights'])],mask,records)
                    layers.append(a)
                    for v in range(u+1,self.U):
                        self.prefix[v].append(token(7,[u/max(1,self.U-1),a/7]))
                self._apply(dict(layers=layers))
            elif kind == 'commitments':
                decisions={}
                for req in event['requests']:
                    u=req['uav']; lo=req['earliest']; hi=req['latest']
                    if req['keep']:
                        lo=min(lo,req['keep_T']); hi=max(hi,req['keep_T'])
                    legal_times=self._legal_times(req,lo,hi)
                    if self.flags.f1:
                        times = (lo+np.flatnonzero(legal_times)).tolist()
                        features = [[T/self.N, (T-lo)/self.N,
                                     *(self.request_features(req,T) if self.flags.f6 else []),
                                     *(self.readiness_candidate(req,T) if self.flags.f15 else [])] for T in times]
                        a = self._choose(actor,u,'timing',features,[True]*len(times),records,req)
                        T = times[a]
                    else:
                        W=hi-lo+1; nb=min(16,W)
                        bins=[(b*W//nb,(b+1)*W//nb) for b in range(nb)]
                        b=self._choose(actor,u,'bin',[[b/16,(lo+l)/self.N,(lo+h-1)/self.N,
                            *(self.request_features(req,lo+l) if self.flags.f6 else [])] for b,(l,h) in enumerate(bins)],
                                       [any(legal_times[l:h]) for l,h in bins],records,req)
                        l,h=bins[b]
                        off=self._choose(actor,u,'offset',[[o/max(1,h-l),(lo+l+o)/self.N,
                            *(self.request_features(req,lo+l+o) if self.flags.f6 else [])] for o in range(h-l)],
                                         legal_times[l:h],records,req)
                        T=lo+l+off
                    opts=list(req['routes'])
                    features=[[r['route']/4,r['travel_slots']/self.N,(r['energy_J'] or 0)/ex.state.p['battery'],T/self.N]
                              for r in opts]
                    mask=[r['legal'] and r['earliest']<=T<=r['latest'] for r in opts]
                    features.append([-1,0,0,T/self.N]); mask.append(req['keep'] and T==req['keep_T'])
                    a=self._choose(actor,u,'route',features,mask,records,req)
                    decisions[u]=dict(T=T,route='keep' if a==len(opts) else opts[a]['route'])
                self._apply(dict(commitments=decisions))
                if self.flags.f2:
                    self.deadline_failures()
            elif kind == 'locations':
                decisions={}
                for k in event['tasks']:
                    opts=event['legal_locations'][k]; u=ex.state.tasks[k]['u']
                    features = []
                    for m in opts:
                        row = [m/(ex.state.p['M']+1),k/self.K]
                        if self.flags.f6:
                            bs = self._bs[m-1] if m else ex.state.q[u]
                            los = m > 0 and not any(segment_box(ex.state.q[u],bs,b) for b in self._buildings)
                            row += [float(m==0), *(bs-ex.state.q[u])/1000, float(los),
                                    0 if m==0 else self.instance['load'][m-1][n]/ex.state.p['G_total'],
                                    sum(float(st.W[1]) for j,st in enumerate(self.task_states())
                                        if st.location==m and (m!=0 or ex.state.tasks[j]['u']==u))/1e10]
                        features.append(row)
                    if self.flags.f16:
                        features = [[*row, *values] for row, values in zip(features, self.location_readiness(k, opts))]
                    a=self._choose(actor,u,'location',features,[True]*len(opts),records)
                    decisions[k]=opts[a]
                self._apply(dict(locations=decisions))
            elif kind == 'priorities':
                priorities=np.zeros(event['shape'])
                contended = self.contended(event) if self.flags.f7 else None
                for k,i in event['active']:
                    if contended is not None and (k,i) not in contended:
                        priorities[k,i] = 1.
                        continue
                    u=ex.state.tasks[k]['u']
                    a=self._choose(actor,u,'priority',[[level/16,k/self.K,i/2,w/16] for level,w in enumerate(PRIORITIES)],
                                   [w>=event['minimum'] for w in PRIORITIES],records)
                    priorities[k,i]=PRIORITIES[a]
                self._apply(dict(priorities=priorities.tolist()))
                history=ex.state.history
                if self.flags.f2 or self.flags.f4:
                    self.audit_features.add(thaw(history.slot))
                for k, caps in enumerate(history.capacities):
                    for i,c in enumerate(caps):
                        self.capacity_history[n,k,i]=[float(c.lo)/1e10,float(c.hi)/1e10]
                self.frequency_history[n]=np.asarray(history.slot['f'])/1e10
            else:
                raise RuntimeError(f'unexpected event {kind}')
        reward=self.phi()-before
        if ex.state.n == self.N:
            self.last_result=ex.result()
            self.last_result['metrics']=terminal_metrics(self.last_result,self.instance)
            reward-=self.last_result['metrics']['loss']
        if self.flags.f14c:
            self._observed_reward = self._observed_reward+reward if ex.state.n < self.N else 0.
        return records,float(reward),ex.state.n==self.N
