import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import TOKEN_DIM, CHOICE_DIM, HEADS, ALL_HEADS
from .flags import flags_or_default
from .official_mlp import MLPBase
from types import SimpleNamespace


def hazard_log_probs(logits, legal):
    legal = torch.as_tensor(legal, dtype=torch.bool, device=logits.device)
    if not legal.any(dim=-1).all():
        raise ValueError('empty hazard support')
    last = legal & (legal.to(torch.int64).flip(-1).cumsum(-1).flip(-1) == 1)
    stay = torch.nn.functional.logsigmoid(-logits).masked_fill(~legal | last, 0.)
    prefix = stay.cumsum(-1)-stay
    stop = torch.nn.functional.logsigmoid(logits).masked_fill(last, 0.)
    return (prefix+stop).masked_fill(~legal, -torch.inf)


def init(module, gain=math.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


class AttentionBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.qkv=nn.Linear(width,3*width)
        self.proj=nn.Linear(width,width)
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
        self.heads=heads

    def forward(self,x,valid):
        B,L,D=x.shape
        q,k,v=self.qkv(x).reshape(B,L,3,self.heads,D//self.heads).permute(2,0,3,1,4)
        attn=torch.nn.functional.scaled_dot_product_attention(q,k,v,
                    attn_mask=valid[:,None,None,:],dropout_p=0.)
        x=self.norm1(x+self.proj(attn.transpose(1,2).reshape(B,L,D)))
        return self.norm2(x+self.ff(x))


class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(TOKEN_DIM),nn.Linear(TOKEN_DIM,128),nn.ReLU())
        self.layers=nn.ModuleList([AttentionBlock(),AttentionBlock()])
        self.apply(init)

    def forward(self,tokens,valid):
        x=self.input(tokens)
        for layer in self.layers:
            x=layer(x,valid)
        return x[:,0]


def pad_observations(observations,device):
    size=max(len(o) for o in observations)
    x=np.zeros((len(observations),size,TOKEN_DIM),np.float32)
    mask=np.zeros((len(observations),size),bool)
    for i,o in enumerate(observations):
        x[i,:len(o)]=o; mask[i,:len(o)]=True
    return torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device)


class Actor(nn.Module):
    def __init__(self, flags=None):
        super().__init__()
        self.flags = flags_or_default(flags)
        self.encoder=EntityEncoder()
        heads = ALL_HEADS if self.flags.f1 else HEADS
        self.heads=nn.ModuleDict({h:nn.Sequential(nn.Linear(128+CHOICE_DIM,128),nn.ReLU(),nn.Linear(128,1)) for h in heads})
        for head in self.heads.values():
            head.apply(init); init(head[-1],.01)
        if self.flags.f14a:
            from .f14_candidate import CandidatePointer
            with torch.random.fork_rng(devices=[]):
                self.candidate_heads=nn.ModuleDict({h:CandidatePointer(CHOICE_DIM) for h in heads})

    def distribution(self,head,observations,candidates,masks):
        device=next(self.parameters()).device
        obs,valid=pad_observations(observations,device)
        context=self.encoder(obs,valid)
        width=max(len(c) for c in candidates)
        data=np.zeros((len(candidates),width,CHOICE_DIM),np.float32)
        legal=np.zeros((len(candidates),width),bool)
        for i,(c,m) in enumerate(zip(candidates,masks)):
            data[i,:len(c)]=c; legal[i,:len(c)]=m
        if not legal.any(axis=1).all():
            raise ValueError('empty mask')
        data=torch.as_tensor(data,device=device)
        logits=self.heads[head](torch.cat([context[:,None].expand(-1,width,-1),data],-1)).squeeze(-1)
        if self.flags.f14a:
            logits=logits+self.candidate_heads[head](context,data,legal).squeeze(-1)
        if head == 'timing':
            return Categorical(logits=hazard_log_probs(logits, legal))
        logits=logits.masked_fill(~torch.as_tensor(legal,device=device),-torch.inf)
        return Categorical(logits=logits)

    def record_distribution(self,records):
        if not records or len({r.head for r in records}) != 1:
            raise ValueError('evaluate one nonempty head family per tensor batch')
        return self.distribution(records[0].head,[r.observation for r in records],
                                 [r.candidates for r in records],[r.mask for r in records])

    def evaluate(self,records):
        dist=self.record_distribution(records)
        actions=torch.tensor([r.action for r in records],device=dist.logits.device)
        return dist.log_prob(actions),dist.entropy()


class ActorBank(nn.Module):
    def __init__(self,uavs,shared=True,flags=None):
        super().__init__()
        self.uavs=uavs; self.shared=shared
        self.flags = flags_or_default(flags)
        self.actors=nn.ModuleList([Actor(self.flags) for _ in range(1 if shared else uavs)])

    def for_uav(self,u):
        return self.actors[0 if self.shared else u]

    @torch.inference_mode()
    def sample(self,u,head,observation,candidates,mask):
        dist=self.for_uav(u).distribution(head,[observation],[candidates],[mask])
        action=dist.sample()
        return int(action.item()),float(dist.log_prob(action).item())


class Critic(nn.Module):
    def __init__(self,state_dim,uavs,local=False,flags=None):
        super().__init__()
        self.state_dim=state_dim; self.uavs=uavs; self.local=local
        self.flags = flags_or_default(flags)
        if local:
            self.encoder=EntityEncoder()
            if self.flags.f3:
                self.base = MLPBase(SimpleNamespace(use_feature_normalization=True, use_orthogonal=True,
                    use_ReLU=True, stacked_frames=1, layer_N=1, hidden_size=64), (128,))
            self.value=nn.Linear(64 if self.flags.f3 else 128,1); init(self.value,1.)
        elif self.flags.f3:
            base = MLPBase(SimpleNamespace(use_feature_normalization=True, use_orthogonal=True,
                use_ReLU=True, stacked_frames=1, layer_N=1, hidden_size=64), (state_dim,))
            value = nn.Linear(64,1); init(value,1.)
            self.net = nn.Sequential(base, value)
        else:
            self.net=nn.Sequential(nn.LayerNorm(state_dim),nn.Linear(state_dim,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
            self.net.apply(init); init(self.net[-1],1.)

    def forward(self,states):
        device=next(self.parameters()).device
        if self.local:
            obs,valid=pad_observations(states,device)
            features = self.encoder(obs,valid)
            return self.value(self.base(features) if self.flags.f3 else features).squeeze(-1)
        arr=np.zeros((len(states),self.state_dim),np.float32)
        for i,s in enumerate(states):
            if len(s)>self.state_dim:
                raise ValueError('state dimension exceeds frozen pool schema')
            arr[i,:len(s)]=s
        return self.net(torch.as_tensor(arr,device=device)).squeeze(-1)


# Adapted from marlbenchmark/on-policy (commit de66d7a), onpolicy/utils/valuenorm.py; MIT License, see LICENSE.on-policy.
class ValueNorm(nn.Module):
    def __init__(self,beta=.99999,epsilon=1e-5):
        super().__init__(); self.beta=beta; self.epsilon=epsilon
        self.register_buffer('running_mean',torch.zeros(()))
        self.register_buffer('running_mean_sq',torch.zeros(()))
        self.register_buffer('debiasing_term',torch.zeros(()))

    def moments(self):
        den=self.debiasing_term.clamp(min=self.epsilon)
        mean=self.running_mean/den
        var=(self.running_mean_sq/den-mean.square()).clamp(min=1e-2)
        return mean,var

    @torch.no_grad()
    def update(self,x):
        x=torch.as_tensor(x,device=self.running_mean.device,dtype=torch.float32)
        self.running_mean.mul_(self.beta).add_(x.mean()*(1-self.beta))
        self.running_mean_sq.mul_(self.beta).add_(x.square().mean()*(1-self.beta))
        self.debiasing_term.mul_(self.beta).add_(1-self.beta)

    def normalize(self,x):
        m,v=self.moments(); return (x-m)/v.sqrt()

    def denormalize(self,x):
        m,v=self.moments(); return x*v.sqrt()+m
