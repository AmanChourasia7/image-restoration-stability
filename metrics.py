import torch
import torch.nn.functional as F
import math


def mse(pred, target):
    return F.mse_loss(pred, target)


def psnr(pred, target):

    mse_value = F.mse_loss(pred, target)

    if mse_value == 0:
        return 100

    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse_value))

    return psnr_value
