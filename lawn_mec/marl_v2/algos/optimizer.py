from typing import Protocol, Callable
import torch


class OptimizerStep(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> float: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class AdamStep:
    def __init__(self,parameters,*,lr=5e-4,eps=1e-5,max_grad_norm=10.):
        self.parameters=list(parameters); self.max_grad_norm=max_grad_norm
        self.optimizer=torch.optim.Adam(self.parameters,lr=lr,eps=eps,weight_decay=0.)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def step(self):
        grad=torch.nn.utils.clip_grad_norm_(self.parameters,self.max_grad_norm,error_if_nonfinite=True)
        self.optimizer.step()
        return float(grad)


FACTORIES: dict[str,Callable[...,OptimizerStep]]={'adam':AdamStep}


def register_optimizer(name,factory):
    if name in FACTORIES:
        raise ValueError(f'optimizer already registered: {name}')
    FACTORIES[name]=factory
