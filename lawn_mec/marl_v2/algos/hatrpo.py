import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.attention import sdpa_kernel, SDPBackend


def flat_parameters(module):
    return parameters_to_vector(module.parameters()).detach().clone()


def _flatten(grads, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g,p in zip(grads,parameters)])


def hessian_vector_product(loss, parameters, vector):
    parameters = tuple(parameters)
    first = torch.autograd.grad(loss,parameters,create_graph=True,allow_unused=True)
    directional = (_flatten(first,parameters)*vector).sum()
    if not directional.requires_grad:
        return torch.zeros_like(vector)
    second = torch.autograd.grad(directional,parameters,allow_unused=True)
    return _flatten(second,parameters).detach()


def conjugate_gradient(product, rhs, steps=10, residual_tol=1e-10):
    solution = torch.zeros_like(rhs)
    residual = rhs.detach().clone()
    direction = residual.clone()
    norm_sq = torch.dot(residual,residual)
    if float(norm_sq) == 0.:
        return solution
    for _ in range(steps):
        image = product(direction)
        curvature = torch.dot(direction,image)
        if not torch.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError('nonpositive/nonfinite damped Fisher curvature')
        rate = norm_sq/curvature
        solution = solution + rate*direction
        residual = residual-rate*image
        next_norm_sq = torch.dot(residual,residual)
        if float(next_norm_sq) < residual_tol:
            break
        direction = residual + (next_norm_sq/norm_sq)*direction
        norm_sq = next_norm_sq
    return solution


def categorical_kl(old_log_probs, old_probs, new_log_probs):
    legal = torch.isfinite(old_log_probs)
    old = old_log_probs.masked_fill(~legal,0.)
    new = new_log_probs.masked_fill(~legal,0.)
    return (old_probs*(old-new)).sum(-1)


def trust_region_step(trainer, batch, advantage, *, target, factor):
    cfg = trainer.cfg
    actor = trainer.actors.for_uav(target)
    parameters = tuple(actor.parameters())
    device = trainer.device
    active = torch.as_tensor(batch.active[:,target],device=device)
    count = int(active.sum())
    row = dict(agent=int(target),accepted=False,line_search_index=-1,kl=0.,
               expected_improvement=0.,actual_improvement=0.,trial_kl=0.,
               trial_actual_improvement=0.,policy_loss=0.,entropy=0.,actor_grad=0.,
               reason='inactive')
    if count == 0:
        return row
    weights = torch.as_tensor(factor[:,target]*advantage[:,target],device=device)*active/count
    old_rollout = torch.as_tensor(batch.old_logp[:,target],device=device)
    groups = []
    initial_logp = torch.zeros(batch.T,device=device)
    entropy = torch.zeros((),device=device)
    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            for _, ids, records in trainer._groups(batch,target):
                slots = torch.as_tensor(batch.indices[ids]//batch.U,device=device)
                dist = actor.record_distribution(records)
                actions = torch.tensor([r.action for r in records],device=device)
                initial_logp.index_add_(0,slots,dist.log_prob(actions))
                entropy += dist.entropy().sum()
                groups.append((records,slots,actions,dist.logits.detach(),dist.probs.detach()))
            initial_ratio = (initial_logp-old_rollout).exp()
            initial_objective = (weights*initial_ratio).sum()
            coefficient = weights*initial_ratio
        gradient = torch.zeros_like(flat_parameters(actor))
        for records, slots, actions, _, _ in groups:
            dist = actor.record_distribution(records)
            term = (dist.log_prob(actions)*coefficient[slots]).sum()
            grads = torch.autograd.grad(term,parameters,allow_unused=True)
            gradient += _flatten(grads,parameters).detach()
        row.update(policy_loss=-float(initial_objective),entropy=float(entropy/count),
                   actor_grad=float(torch.linalg.vector_norm(gradient)),reason='zero_gradient')
        if not torch.isfinite(gradient).all() or not torch.isfinite(initial_objective):
            raise FloatingPointError('nonfinite HATRPO surrogate/gradient')
        if not torch.count_nonzero(gradient):
            return row

        def fisher(vector):
            result = torch.zeros_like(vector)
            for records, _, _, old_log, old_probs in groups:
                dist = actor.record_distribution(records)
                kl = categorical_kl(old_log,old_probs,dist.logits).sum()/count
                result += hessian_vector_product(kl,parameters,vector)
            return result + cfg.fisher_damping*vector

        direction = conjugate_gradient(fisher,gradient,cfg.cg_steps,cfg.cg_residual_tol)
        quadratic = torch.dot(direction,fisher(direction))
        if not torch.isfinite(quadratic) or quadratic <= 0:
            row['reason'] = 'invalid_direction'
            return row
        displacement = direction*torch.sqrt(2*cfg.kl_threshold/quadratic)
        predicted = float(torch.dot(gradient,displacement))
        if not math.isfinite(predicted) or predicted <= 0:
            row['reason'] = 'invalid_direction'
            return row
        initial_parameters = flat_parameters(actor)
        row['reason'] = 'line_search_rejected'
        try:
            with torch.no_grad():
                for index in range(cfg.ls_step):
                    fraction = cfg.backtrack_coeff**index
                    vector_to_parameters(initial_parameters+fraction*displacement,parameters)
                    current_logp = torch.zeros_like(initial_logp)
                    slot_kl = torch.zeros_like(initial_logp)
                    for records, slots, actions, old_log, old_probs in groups:
                        dist = actor.record_distribution(records)
                        current_logp.index_add_(0,slots,dist.log_prob(actions))
                        slot_kl.index_add_(0,slots,categorical_kl(old_log,old_probs,dist.logits))
                    objective = (weights*(current_logp-old_rollout).exp()).sum()
                    measured_kl = float(slot_kl.sum()/count)
                    actual = float(objective-initial_objective)
                    expected = fraction*predicted
                    row.update(line_search_index=index,expected_improvement=expected,
                               trial_kl=measured_kl,trial_actual_improvement=actual)
                    if (math.isfinite(measured_kl) and math.isfinite(actual)
                            and measured_kl <= cfg.kl_threshold and actual > 0
                            and actual/expected > cfg.accept_ratio):
                        row.update(accepted=True,kl=measured_kl,actual_improvement=actual,
                                   policy_loss=-float(objective),reason='accepted')
                        break
        finally:
            if not row['accepted']:
                with torch.no_grad():
                    vector_to_parameters(initial_parameters,parameters)
    return row
