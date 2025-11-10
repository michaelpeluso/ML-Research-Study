import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim import Optimizer
from typing import Any, Optional, Tuple

from utils.logger import print_t as print


# sgd with momentum
class SGDMomentum(SGD):
    def __init__(self, params, lr: float,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0,
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):
        super().__init__(params, lr=lr, momentum=0.9, dampening=dampening,
                         weight_decay=weight_decay, nesterov=False,
                         maximize=maximize, foreach=foreach,
                         differentiable=differentiable, fused=fused)
        

# adam without bias correction
class AdamNoBiasCorrection(Adam):
    def __init__(self, params, lr: float,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad,
                         maximize=maximize, foreach=foreach,
                         capturable=capturable, differentiable=differentiable,
                         fused=fused)

    def step(self, closure=None):
        # custom step without bias correction
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                # state access
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # unpack
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                max_exp_avg_sq = state.get('max_exp_avg_sq') if group['amsgrad'] else None
                beta1, beta2 = group['betas']

                # increment step
                state['step'] += 1

                # decay if weight_decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad'] and max_exp_avg_sq is not None:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # no bias correction: use raw emas
                step_size = group['lr']
                # update param
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# implementation redacted from previous report
class SGDNoMomentum(SGD): pass
class NesterovMomentum(SGD): pass
class BaselineAdam(Adam): pass
class AdamRMSPropLike(Adam): pass
class AdamWDecoupled(AdamW): pass


def optimizer_factory(model: torch.nn.Module, kind: str, lr: float = 1e-3, **kwargs: Any) -> Optional[Optimizer]:
    # filter trainable params
    params = [p for p in model.parameters() if p.requires_grad]

    if kind in ['sgd', 'sgd_momentum', 'nesterov']:
        # extract sgd-specific kwargs with types
        weight_decay = float(kwargs.get('weight_decay', 0.0)) # type: ignore
        dampening = float(kwargs.get('dampening', 0.0)) # type: ignore
        maximize = bool(kwargs.get('maximize', False))
        foreach = kwargs.get('foreach', None)
        differentiable = bool(kwargs.get('differentiable', False))
        fused = kwargs.get('fused', None)

        if kind == 'sgd':
            print("Not implemented. Review OL report.")
            raise NotImplementedError("SGD without momentum is not implemented.")

        elif kind == 'sgd_momentum':
            return SGDMomentum(params, lr=lr, weight_decay=weight_decay, dampening=dampening, maximize=maximize, 
                               foreach=foreach, differentiable=differentiable, fused=fused) # type: ignore
        elif kind == 'nesterov':
            print("Not implemented. Review OL report.")
            raise NotImplementedError("Nesterov momentum is not implemented.")
    
    elif kind in ['adam', 'adam_no_bias', 'rmsprop_like', 'adamw']:
        # extract adam-specific kwargs with types    
        betas = kwargs.get('betas', (0.9, 0.999))
        if not isinstance(betas, tuple) or len(betas) != 2 or not all(isinstance(b, float) for b in betas):
            raise ValueError(f"betas must be tuple of two floats, got {betas}")
        eps = float(kwargs.get('eps', 1e-8)) # type: ignore
        weight_decay = float(kwargs.get('weight_decay', 0.01 if kind == 'adamw' else 0.0)) # type: ignore
        amsgrad = bool(kwargs.get('amsgrad', False))
        maximize = bool(kwargs.get('maximize', False))
        foreach = bool(kwargs.get('foreach', None))
        capturable = bool(kwargs.get('capturable', False))
        differentiable = bool(kwargs.get('differentiable', False))
        fused = bool(kwargs.get('fused', None))

        if kind == 'adam':
            print("Not implemented. Review OL report.")
            raise NotImplementedError("Adam is not implemented.")
        
        elif kind == 'adam_no_bias':
            return AdamNoBiasCorrection(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, 
                                        foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
        elif kind == 'rmsprop_like':
            print("Not implemented. Review OL report.")
            raise NotImplementedError("RMSProp-like optimizer is not implemented.")
        
        elif kind == 'adamw':
            print("Not implemented. Review OL report.")
            raise NotImplementedError("AdamW is not implemented.")
    else:
        raise ValueError(f"Unknown optimizer kind: {kind}")