import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import torch
from nest import register


@register
def cross_entropy_loss(
    input: Tensor, 
    target: Tensor,
    weight: Optional[List[float]] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    """Cross entropy loss.
    """

    return F.cross_entropy(input, target, torch.cuda.FloatTensor(weight), size_average, ignore_index, reduce)


@register
def multilabel_soft_margin_loss(
    input: Tensor, 
    target: Tensor,
    weight: Optional[List[float]] = None,
    size_average: bool = True,
    reduce: bool = True,
    difficult_samples: bool = False) -> Tensor:
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target

    if weight is not None:
        weight = torch.cuda.FloatTensor(weight)
        
    return F.multilabel_soft_margin_loss(input, gt_label, weight, size_average, reduce)
