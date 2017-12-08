"""
@author: Markus Beissinger
University of Pennsylvania, 2017

This class produces the Untied GSN model discussed in the paper: (thesis link)
"""
import torch.nn as nn

from src.models.gsn import GSN


class UntiedGSN(nn.Module):
    def __init__(self, sizes, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True):
        super().__init__()
        # make our GSN
        self.gsn = GSN(
            sizes=sizes, tied_weights=False, visible_act=visible_act, hidden_act=hidden_act,
            input_noise=input_noise, hidden_noise=hidden_noise, input_sampling=input_sampling, noiseless_h1=noiseless_h1
        )

    def forward(self, xs=None):
        predictions = []
        if xs is not None:
            hiddens = self.gsn.init_hiddens(xs[0])
            for x in xs:
                hiddens = self.gsn.encode(x, hiddens)
                x_recon, hiddens, _ = self.gsn.decode(hiddens)
                predictions.append(x_recon)

        return predictions