import weakref
import numpy as np
import torch
from torch import nn


class PotentialResidualCritic(nn.Module):
    def __init__(self, base, normalizer):
        super().__init__()
        self.base=base
        self._normalizer=weakref.ref(normalizer)
        self.state_dim=base.state_dim; self.uavs=base.uavs; self.local=base.local
        self.flags=base.flags

    def forward(self, states):
        if self.local:
            phi=np.asarray([s[-1,8] for s in states],np.float32)
            inputs=[s[:-1] for s in states]
        else:
            phi=np.asarray([s[-1] for s in states],np.float32)
            inputs=[s[:-1] for s in states]
        pred=self.base(inputs)
        _,variance=self._normalizer().moments()
        return pred-torch.as_tensor(phi,device=pred.device)/variance.sqrt()


def output_layer(critic):
    if isinstance(critic,PotentialResidualCritic):
        critic=critic.base
    return critic.value if critic.local else critic.net[-1]


@torch.no_grad()
def update_value_coordinates(critic, normalizer, returns):
    old_mean,old_var=normalizer.moments()
    normalizer.update(returns)
    mean,var=normalizer.moments()
    scale=old_var.sqrt()/var.sqrt()
    layer=output_layer(critic)
    layer.weight.mul_(scale)
    layer.bias.mul_(scale).add_((old_mean-mean)/var.sqrt())
