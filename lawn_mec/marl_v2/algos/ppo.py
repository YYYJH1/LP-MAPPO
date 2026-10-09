# A2PO components adapted from xihuai18/A2PO-ICLR2023 (commit 28c11e6); MIT License, see lawn_mec/marl_v2/LICENSE.A2PO.
from dataclasses import dataclass,asdict
from collections import defaultdict
import numpy as np
import torch

from ..batch import normalize_advantage
from ..nets import ActorBank,Critic,ValueNorm
from .optimizer import FACTORIES
from ..flags import flags_or_default, flags_record

DISPLAY_NAMES={'mappo':'MAPPO','happo':'HAPPO','ippo':'IPPO',
               'coppo':'CoPPO-style joint-ratio PPO','a2po':'A2PO','hatrpo':'HATRPO'}


@dataclass(frozen=True)
class Config:
    lr: float=5e-4
    critic_lr: float=5e-4
    adam_eps: float=1e-5
    epochs: int=5
    minibatches: int=1
    clip: float=.2
    entropy: float=.01
    gae_lambda: float=.95
    gamma: float=1.
    max_grad_norm: float=10.
    huber_delta: float=10.
    others_clip: float=.1
    a2po_lambda: float=.93
    head_chunk: int=128
    value_chunk: int=128


@dataclass(frozen=True)
class HATRPOConfig(Config):
    entropy: float=0.
    kl_threshold: float=.01
    ls_step: int=10
    accept_ratio: float=.5
    backtrack_coeff: float=.8
    cg_steps: int=10
    cg_residual_tol: float=1e-10
    fisher_damping: float=.1


def rp_clips(ratios,clip=.2):
    r=np.abs(1-np.asarray(ratios))+1
    suffix=np.cumprod(r[:,::-1],axis=1)[:,::-1]
    weight=np.mean(1/suffix,axis=0)
    return weight/weight.max()*clip


def semi_greedy(advantages,old_normalized_values,rng):
    score=np.mean(np.abs(advantages/np.maximum(np.abs(old_normalized_values),1e-8)),axis=0)
    remaining=list(np.argsort(-score,kind='stable')); order=[]
    while remaining:
        order.append(remaining.pop(0))
        if remaining:
            order.append(remaining.pop(int(rng.integers(len(remaining)))))
    return [int(u) for u in order]


def surrogate(logp,old,adv,kind,clip=.2,others_clip=.1,factor=None,target=None):
    ratio=(logp-old).exp()
    if kind in ('mappo','ippo'):
        return torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
    if kind=='happo':
        return factor*torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
    other=[]
    for u in range(ratio.shape[1]):
        indices=[v for v in range(ratio.shape[1]) if v!=u]
        other.append(ratio[:,indices].detach().prod(1) if indices else torch.ones_like(ratio[:,u]))
    other=torch.stack(other,1).clamp(1-others_clip,1+others_clip)
    effective=ratio*other
    return torch.minimum(effective*adv,torch.maximum(torch.minimum(effective,1+clip),1-clip)*adv)


class Trainer:
    def __init__(self,algorithm,state_dim,uavs,*,device='cpu',seed=200,config=None,optimizer='adam',flags=None):
        if algorithm not in DISPLAY_NAMES:
            raise ValueError(algorithm)
        self.algorithm=algorithm; self.cfg=config or Config(); self.device=torch.device(device)
        if algorithm=='hatrpo':
            self.cfg=HATRPOConfig(**{**asdict(self.cfg),'entropy':0.})
        self.flags = flags_or_default(flags)
        if self.cfg.gamma!=1 or self.cfg.minibatches!=1:
            raise ValueError('training requires gamma=1 and one full minibatch')
        torch.manual_seed(seed); self.rng=np.random.default_rng(seed)
        self.actors=ActorBank(uavs,shared=algorithm not in ('happo','hatrpo'),flags=self.flags).to(self.device)
        self.critic=Critic(state_dim,uavs,local=algorithm=='ippo',flags=self.flags).to(self.device)
        self.normalizer=ValueNorm().to(self.device)
        if self.flags.f14c:
            from ..f14_value import PotentialResidualCritic
            self.critic=PotentialResidualCritic(self.critic,self.normalizer)
        kwargs=dict(eps=self.cfg.adam_eps,max_grad_norm=self.cfg.max_grad_norm)
        self.actor_opts=([] if algorithm=='hatrpo' else
                         [FACTORIES[optimizer](a.parameters(),lr=self.cfg.lr,**kwargs) for a in self.actors.actors])
        self.critic_opt=FACTORIES[optimizer](self.critic.parameters(),lr=self.cfg.critic_lr,**kwargs)

    def description(self):
        return dict(algorithm=DISPLAY_NAMES[self.algorithm],config=asdict(self.cfg),flags=flags_record(self.flags),
                    shared_actor=self.actors.shared,local_critic=self.critic.local,
                    actor_parameters=sum(p.numel() for p in self.actors.parameters()),
                    critic_parameters=sum(p.numel() for p in self.critic.parameters()))

    def _groups(self,batch,target=None):
        groups=defaultdict(list)
        for i,r in enumerate(batch.records):
            if target is None or r.uav==target:
                groups[(0 if self.actors.shared else r.uav,r.head)].append(i)
        for (agent,head),indices in groups.items():
            for start in range(0,len(indices),self.cfg.head_chunk):
                ids=indices[start:start+self.cfg.head_chunk]
                yield self.actors.actors[agent],ids,[batch.records[i] for i in ids]

    @torch.no_grad()
    def log_probabilities(self,batch):
        logp=torch.zeros(batch.T*batch.U,device=self.device)
        entropy=torch.zeros_like(logp)
        for actor,ids,records in self._groups(batch):
            lp,en=actor.evaluate(records)
            index=torch.as_tensor(batch.indices[ids],device=self.device)
            logp.index_add_(0,index,lp); entropy.index_add_(0,index,en)
        return logp.reshape(batch.T,batch.U),entropy.reshape(batch.T,batch.U)

    def _actor_step(self,batch,adv,*,target=None,factor=None,clip=None,logp=None):
        cfg=self.cfg
        lp=self.log_probabilities(batch)[0] if logp is None else logp.detach()
        lp.requires_grad_(True)
        old=torch.as_tensor(batch.old_logp,device=self.device)
        adv=torch.as_tensor(adv,device=self.device)
        mask=torch.as_tensor(batch.active,device=self.device)
        if target is not None:
            chosen=torch.zeros_like(mask); chosen[:,target]=mask[:,target]; mask=chosen
        denominator=mask.sum().clamp(min=1)
        eps=torch.as_tensor(cfg.clip if clip is None else clip,device=self.device)
        fac=None if factor is None else torch.as_tensor(factor,device=self.device)
        objective=surrogate(lp,old,adv,self.algorithm,eps,cfg.others_clip,fac)
        loss=-(objective*mask).sum()/denominator
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite policy surrogate')
        derivative=torch.autograd.grad(loss,lp)[0].detach().reshape(-1)
        opts=self.actor_opts if target is None or self.actors.shared else [self.actor_opts[target]]
        for opt in opts:
            opt.zero_grad()
        entropy_total=0.
        for actor,ids,records in self._groups(batch,target):
            head_lp,head_entropy=actor.evaluate(records)
            index=torch.as_tensor(batch.indices[ids],device=self.device)
            term=(head_lp*derivative[index]).sum()-cfg.entropy*head_entropy.sum()/denominator
            term.backward(); entropy_total+=float(head_entropy.detach().sum())
        grads=[opt.step() for opt in opts]
        return dict(policy_loss=float(loss.detach()),entropy=entropy_total/float(denominator),actor_grad=max(grads))

    def _critic_step(self,batch,returns,target=None):
        cfg=self.cfg; device=self.device
        norm_targets=returns[:,target] if target is not None else returns
        if self.flags.f14b:
            from ..f14_value import update_value_coordinates
            update_value_coordinates(self.critic,self.normalizer,norm_targets)
        else:
            self.normalizer.update(norm_targets)
        if self.critic.local:
            inputs=[o for obs in batch.local_states for o in obs]
            old=batch.normalized_values.reshape(-1); targets=returns.reshape(-1)
        else:
            inputs=batch.states
            old=batch.normalized_values[:,0]
            targets=returns.mean(1) if target is None else returns[:,target]
        if self.flags.f14b:
            raw=batch.values.reshape(-1) if self.critic.local else batch.values[:,0]
            old=self.normalizer.normalize(torch.as_tensor(raw,device=device)).detach()
        self.critic_opt.zero_grad(); total=0.
        for start in range(0,len(inputs),cfg.value_chunk):
            end=start+cfg.value_chunk
            pred=self.critic(inputs[start:end])
            anchor=torch.as_tensor(old[start:end],device=device)
            truth=self.normalizer.normalize(torch.as_tensor(targets[start:end],device=device))
            clipped=anchor+(pred-anchor).clamp(-cfg.clip,cfg.clip)
            def huber(error):
                a=error.abs(); q=a.clamp(max=cfg.huber_delta)
                return .5*q.square()+cfg.huber_delta*(a-q)
            loss=torch.maximum(huber(truth-pred),huber(truth-clipped)).sum()/len(inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite value loss')
            loss.backward(); total+=float(loss.detach())
        grad=self.critic_opt.step()
        return dict(value_loss=total,critic_grad=grad)

    @torch.no_grad()
    def _a2po_stored_values(self,batch):
        if self.flags.f14b:
            return batch.values.copy()
        stored=torch.as_tensor(batch.normalized_values,device=self.device)
        return self.normalizer.denormalize(stored).cpu().numpy()

    def _hatrpo_step(self,batch,adv,*,target,factor):
        from .hatrpo import trust_region_step
        return trust_region_step(self,batch,adv,target=target,factor=factor)

    def update(self,batch):
        if batch.U!=self.actors.uavs:
            raise ValueError('agent count mismatch')
        self.actors.train(); self.critic.train()
        lam=self.cfg.a2po_lambda if self.algorithm=='a2po' else self.cfg.gae_lambda
        advantage,returns=batch.advantages(lam)
        adv=normalize_advantage(advantage,batch.active)
        stats=[]; orders=[]; clips=[]; trust_region=[]
        if self.algorithm in ('happo','hatrpo'):
            factor=np.ones_like(adv)
            order=self.rng.permutation(batch.U).tolist(); orders.append(order)
            for u in order:
                before=self.log_probabilities(batch)[0].cpu().numpy()[:,u]
                if self.algorithm=='hatrpo':
                    step=self._hatrpo_step(batch,adv,target=u,factor=factor)
                    trust_region.append(step)
                    stats.append({k:step[k] for k in ('policy_loss','entropy','actor_grad')})
                else:
                    for _ in range(self.cfg.epochs):
                        stats.append(self._actor_step(batch,adv,target=u,factor=factor))
                after=self.log_probabilities(batch)[0].cpu().numpy()[:,u]
                factor*=np.exp(after-before)[:,None]
            for _ in range(self.cfg.epochs):
                stats.append(self._critic_step(batch,returns))
        elif self.algorithm=='a2po':
            sequence_adv=adv.copy()
            for epoch in range(self.cfg.epochs):
                current_logp=self.log_probabilities(batch)[0]
                ratios=np.exp(current_logp.cpu().numpy()-batch.old_logp)
                eps=rp_clips(ratios,self.cfg.clip); clips.append(eps.tolist())
                if epoch:
                    joint=ratios.prod(1,keepdims=True).repeat(batch.U,1)
                    current_values=self._a2po_stored_values(batch)
                    advantage,returns=batch.advantages(lam,values=current_values,joint_ratios=joint,speedup=True)
                    adv=normalize_advantage(advantage,batch.active)
                order=semi_greedy(sequence_adv,batch.normalized_values,self.rng); orders.append(order)
                for position,u in enumerate(order):
                    if position:
                        current_logp=self.log_probabilities(batch)[0]
                        ratios=np.exp(current_logp.cpu().numpy()-batch.old_logp)
                        joint=ratios.prod(1,keepdims=True).repeat(batch.U,1)
                        current_values=self._a2po_stored_values(batch)
                        updated,_=batch.advantages(lam,values=current_values,joint_ratios=joint,speedup=True)
                        returns[:,u]=updated[:,u]+current_values[:,u]
                        advantage=returns-current_values
                        adv=normalize_advantage(advantage,batch.active)
                    stats.append(self._critic_step(batch,returns,target=u))
                    stats.append(self._actor_step(batch,adv,target=u,clip=eps,logp=current_logp))
                    stats.append(self._critic_step(batch,returns,target=u))
        else:
            for _ in range(self.cfg.epochs):
                stats.append(self._actor_step(batch,adv))
                stats.append(self._critic_step(batch,returns))
        keys=set().union(*(s.keys() for s in stats))
        summary={k:float(np.mean([s[k] for s in stats if k in s])) for k in sorted(keys)}
        if not all(np.isfinite(v) for v in summary.values()):
            raise FloatingPointError(summary)
        summary.update(algorithm=DISPLAY_NAMES[self.algorithm],orders=orders,rp_clips=clips,
                       optimizer_steps=len(stats),episodes=len(batch.episodes))
        if self.algorithm=='hatrpo':
            summary['trust_region']=trust_region
        if self.flags.f8:
            summary.update(self.diagnostics(batch))
        return summary

    @torch.no_grad()
    def diagnostics(self, batch):
        lp, _ = self.log_probabilities(batch)
        logratio = lp.cpu().numpy()-batch.old_logp
        active = logratio[batch.active]
        entropy = defaultdict(list)
        for actor, ids, records in self._groups(batch):
            _, ent = actor.evaluate(records)
            entropy[records[0].head].extend(ent.cpu().tolist())
        inputs = ([o for row in batch.local_states for o in row] if self.critic.local else batch.states)
        pred = torch.cat([self.normalizer.denormalize(self.critic(inputs[i:i+self.cfg.value_chunk]))
                          for i in range(0,len(inputs),self.cfg.value_chunk)]).cpu().numpy()
        pred = pred.reshape(batch.T,batch.U) if self.critic.local else np.repeat(pred[:,None],batch.U,axis=1)
        mc = np.concatenate([np.cumsum(e.rewards[::-1],dtype=np.float64)[::-1] for e in batch.episodes])
        truth = np.repeat(mc[:,None],batch.U,axis=1)
        variance = float(np.var(truth))
        ev = None if variance == 0 else float(1-np.var(truth-pred)/variance)
        starts = np.cumsum([0]+batch.lengths[:-1])
        mean, var = self.normalizer.moments()
        return dict(approx_kl=float(np.mean(np.expm1(active)-active)) if len(active) else 0.,
                    clip_fraction=float(np.mean(np.abs(np.exp(active)-1)>self.cfg.clip)) if len(active) else 0.,
                    per_head_entropy={k:float(np.mean(v)) for k,v in entropy.items()},
                    EV_MC=ev, V0_minus_G0=float(np.mean(pred[starts]-truth[starts])),
                    V0_minus_G0_paired=(pred[starts]-truth[starts]).mean(1).tolist(),
                    value_norm=dict(mean=float(mean),variance=float(var),
                                    debiasing_term=float(self.normalizer.debiasing_term)))

    def checkpoint(self):
        return dict(algorithm=self.algorithm, config=asdict(self.cfg), flags=flags_record(self.flags),
                    state_dim=self.critic.state_dim, U=self.actors.uavs,
                    actors=self.actors.state_dict(), critic=self.critic.state_dict(),
                    normalizer=self.normalizer.state_dict(),
                    actor_optimizers=[o.state_dict() for o in self.actor_opts],
                    critic_optimizer=self.critic_opt.state_dict(),
                    numpy_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None)

    def restore(self, checkpoint):
        if (checkpoint['algorithm'] != self.algorithm or flags_or_default(checkpoint['flags']) != self.flags
                or checkpoint['config'] != asdict(self.cfg)):
            raise ValueError('checkpoint algorithm, flags or hyperparameters differ')
        for name in ('actors','critic','normalizer'):
            getattr(self,name).load_state_dict(checkpoint[name])
        if len(checkpoint['actor_optimizers']) != len(self.actor_opts):
            raise ValueError('optimizer count differs')
        for opt, state in zip(self.actor_opts, checkpoint['actor_optimizers']): opt.load_state_dict(state)
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer'])
        self.rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if self.device.type=='cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
