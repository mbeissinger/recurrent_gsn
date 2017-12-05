import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = Variable(
                torch.Tensor(
                    np.random.normal(loc=self.mean, scale=self.std, size=x.size()).astype('float32')
                ),
                requires_grad=False
            )
            if x.is_cuda:
                noise = noise.cuda()
            return x + noise
        return x


class SaltAndPepper(nn.Module):
    def __init__(self, amount=0.2):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        if self.training:
            a = Variable(
                torch.Tensor(
                    np.random.binomial(n=1, p=1. - self.amount, size=x.size()).astype('float32')
                ),
                requires_grad=False
            )
            b = Variable(
                torch.Tensor(
                    np.random.binomial(n=1, p=0.5, size=x.size()).astype('float32')
                ),
                requires_grad=False
            )
            if x.is_cuda:
                a = a.cuda()
                b = b.cuda()
            c = (a == 0).float() * b
            return x * a + c
        return x
