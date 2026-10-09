import math
import torch
from torch import nn


def relative_candidates(data, legal):
    legal = torch.as_tensor(legal, dtype=torch.bool, device=data.device)
    clean = data.masked_fill(~legal[..., None], 0.)
    count = legal.sum(-1, keepdim=True).clamp(min=1)[..., None]
    mean = clean.sum(-2, keepdim=True)/count
    delta = (clean-mean).masked_fill(~legal[..., None], 0.)
    variance = delta.square().sum(-2, keepdim=True)/count
    relative = delta/variance.clamp(min=1e-8).sqrt()
    return torch.cat((clean, relative), -1)


class CandidatePointer(nn.Module):
    def __init__(self, choice_dim, width=128):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(2*choice_dim,width),nn.ReLU(),nn.LayerNorm(width),
                                 nn.Linear(width,width),nn.ReLU(),nn.LayerNorm(width))
        self.query = nn.Linear(width,width)
        for layer in self.key:
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,math.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.query.weight,.01); nn.init.zeros_(self.query.bias)
        self.scale=math.sqrt(width)

    def forward(self, context, data, legal):
        keys=self.key(relative_candidates(data,legal))
        query=self.query(context)
        return (keys*query[:,None]).sum(-1,keepdim=True)/self.scale
