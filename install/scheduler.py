from typing import Iterable, List

from torch import optim
from nest import register


@register
def step_scheduler(
    optimizer: optim.Optimizer, 
    step_size: int,
    gamma: float = 0.1, 
    last_epoch: int = -1) -> optim.lr_scheduler._LRScheduler:
    """ Step Scheduler.
    """

    return optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)


@register
def multi_step_scheduler(
    optimizer: optim.Optimizer, 
    milestones: List[int],
    gamma: float = 0.1,
    last_epoch: int = -1) -> optim.Optimizer:
    """ Multi Step Scheduler.
    """

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch)
