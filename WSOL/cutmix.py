"""
Original repository: https://github.com/clovaai/CutMix-PyTorch
Reference repository : https://github.com/GenDisc/IVR
"""
import numpy as np
import torch

def cutmix(image, target, beta):
    """
    remove pixels and replace the removed regions with a patch from another image 

    Args:
        image: torch.Tensor, N x C x H x W, float32.
        target: label of the image
        beta: hyperparameter used when sampling from a Beta distribution
    """
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(image.size()[0]).cuda()

    target_a = target.clone().detach()
    target_b = target[rand_index].clone().detach()

    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

    # target_a, target_b will be used for loss computation
    # CrossEntropyLoss(logits, target_a) * lam + 
    # CrossEntropyLoss(logits, target_b) * (1. - lam) 
    return image, target_a, target_b, lam


def rand_bbox(size, lam):
    """
    Args:
        size: (N, C, H, W)
        lam: lambda sampled from Beta distribution
    """
    w = size[3]
    h = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2
