import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    """SGD with cosine annealing.
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (
                    0.5 * (base_lr - self.step_size_min) * (1 + np.cos(self.epochs_since_restart * np.pi / self.t0)))
               for base_lr in self.base_lrs]
        
        return lrs
    
    
def warmup_linear_decay(step, config):
    warm_up_step = config['lr_scheduler']['WarmUpLinearDecay']['warm_up_step']
    train_steps  = config['lr_scheduler']['WarmUpLinearDecay']['train_steps']
    if step < warm_up_step:
        return (step+1)/warm_up_step
    elif step < train_steps:
        return (train_steps-step)/(train_steps-warm_up_step)
    else:
        return 1.0/(train_steps-warm_up_step)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)



def step_lr(step, config):
    milestones  = config['lr_scheduler']['StepLR']['milestones']
    multipliers = config['lr_scheduler']['StepLR']['multipliers']
    n = len(milestones)
    mul = 1
    for i in range(n):
        if step>=milestones[i]:
            mul = multipliers[i]
    return mul