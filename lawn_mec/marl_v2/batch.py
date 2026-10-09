from dataclasses import dataclass
import numpy as np


@dataclass
class Episode:
    records: list
    states: list
    local_states: list
    rewards: np.ndarray
    values: np.ndarray
    normalized_values: np.ndarray
    result: dict
    wall_s: float


class Batch:
    def __init__(self,episodes):
        if not episodes:
            raise ValueError('empty batch')
        self.episodes=episodes
        self.U=episodes[0].values.shape[1]
        self.lengths=[len(e.rewards) for e in episodes]
        self.T=sum(self.lengths)
        self.rewards=np.concatenate([e.rewards for e in episodes])
        self.values=np.concatenate([e.values for e in episodes])
        self.normalized_values=np.concatenate([e.normalized_values for e in episodes])
        self.states=[s for e in episodes for s in e.states]
        self.local_states=[s for e in episodes for s in e.local_states]
        self.old_logp=np.zeros((self.T,self.U),np.float32)
        self.active=np.zeros((self.T,self.U),bool)
        self.records=[]; self.indices=[]
        offset=0
        for e in episodes:
            for record in e.records:
                idx=(offset+record.slot)*self.U+record.uav
                self.records.append(record); self.indices.append(idx)
                self.old_logp.flat[idx]+=record.old_logp
                self.active.flat[idx]=True
            offset+=len(e.rewards)
        self.indices=np.asarray(self.indices,np.int64)

    def advantages(self,lam=.95,values=None,joint_ratios=None,speedup=False):
        values=self.values if values is None else values
        ratios=np.ones_like(values) if joint_ratios is None else joint_ratios
        advantage=np.zeros_like(values); offset=0
        for length in self.lengths:
            v=values[offset:offset+length]
            r=self.rewards[offset:offset+length,None]
            delta=r+np.concatenate([v[1:],np.zeros((1,self.U),np.float32)])-v
            rho=ratios[offset:offset+length]
            if speedup:
                coeff=np.ones((length,self.U)); decay=1.
                accumulated=advantage[offset:offset+length]
                for lag in range(length):
                    size=length-lag
                    if lag:
                        coeff[:size]=np.minimum(1.,coeff[:size]*rho[lag:])
                        decay*=lam
                    accumulated[:size]+=decay*coeff[:size]*delta[lag:]
            else:
                gae=np.zeros(self.U)
                for t in range(length-1,-1,-1):
                    future=np.minimum(1.,rho[t+1]) if t+1<length else np.ones(self.U)
                    gae=delta[t]+lam*future*gae
                    advantage[offset+t]=gae
            offset+=length
        return advantage,advantage+values


def normalize_advantage(advantage,active):
    a=advantage[active]
    return (advantage-a.mean())/(a.std()+1e-5) if len(a) else np.zeros_like(advantage)
