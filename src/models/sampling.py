import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Binomial(nn.Module):
    """
    Similar to implementation https://github.com/Theano/Theano/blob/master/theano/sandbox/rng_mrg.py#L896
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        uses_cuda = x.is_cuda
        p = Variable(x.data, requires_grad=False)
        if uses_cuda:
            p.cuda()
        noise = Variable(
            torch.Tensor(
                np.random.uniform(low=0.0, high=1.0, size=x.size()).astype('float32')
            ),
            requires_grad=False
        )
        if x.is_cuda:
            noise = noise.cuda()

        return (noise < p).float()
