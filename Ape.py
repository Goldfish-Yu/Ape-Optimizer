import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

class Ape(Optimizer):
    def __init__(self, params, lr=required, beta=[0.9, 0.99], power_number=0.9, weight_decay=5e-4, eps=1e-10, fai=10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta[0]))
        if not 0.0 <= beta[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta[1]))
            
        defaults = dict(lr=lr, power_number=power_number, weight_decay=weight_decay, eps=eps, beta=beta, fai=fai)
        super(Ape, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ape, self).__setstate__(state)

    def _init_group(self, group):
        for p in group['params']:
            state = self.state[p]
            if 'I_buffer' not in state:
                state['I_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if 'D_buffer' not in state:
                state['D_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            power_number = group['power_number']
            eps = group['eps']
            beta1 = group['beta'][0]
            beta2 = group['beta'][1]
            fai = group['fai']

            self._init_group(group)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                I_buffer, D_buffer = state['I_buffer'], state['D_buffer']

                D_buffer.mul_(beta2).add_(torch.norm(I_buffer, p=2)**2, alpha=fai)
                I_buffer.mul_(beta1).add_(grad, alpha=1 - beta1)
                p.data.add_(D_buffer * grad / (eps + torch.norm(grad, p=2)**power_number), alpha=-lr)

        return loss
    
