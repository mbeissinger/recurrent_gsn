"""
@author: Markus Beissinger
University of Pennsylvania, 2017

This class produces the RNN-GSN model discussed in the paper: (thesis link)

Inspired by code for the RNN-RBM:
http://deeplearning.net/tutorial/rnnrbm.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from src.models.gsn import GSN
from src.utils import make_time_units_string
from src.data.mnist_sequence import SequencedMNIST

use_cuda = torch.cuda.is_available()


class RNNGSN(nn.Module):
    def __init__(self, sizes, window_size=1, tied_weights=True, walkbacks=5, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True):
        super().__init__()
        self.window_size = window_size
        self.sizes = sizes
        # make our GSN
        self.gsn = GSN(
            sizes=sizes, tied_weights=tied_weights, walkbacks=walkbacks, visible_act=visible_act, hidden_act=hidden_act,
            input_noise=input_noise, hidden_noise=hidden_noise, input_sampling=input_sampling, noiseless_h1=noiseless_h1
        )
        # temporal params, only need one per even layer in the GSN because it alternates layers on updates
        self.regression_weights = []
        self.regression_biases = []
        self.missing_biases = []
        self.regression_parameters = []
        for i, layer_size in enumerate(sizes[1:]):
            if i % 2 == 0:
                weights = []
                missing = []
                for window in range(self.window_size):
                    regression_weight = nn.Parameter(torch.FloatTensor(layer_size, layer_size))
                    # initialize our weight (make sure the gain is calculated for the resulting nonlinearity
                    nn.init.xavier_uniform(tensor=regression_weight, gain=1/self.window_size)  # TODO identity matrix init?
                    self.register_parameter(
                        name='regression_{!s}_{!s}_weight'.format(i+1, "t-{!s}".format(window+1)),
                        param=regression_weight
                    )
                    weights.append(regression_weight)
                    self.regression_parameters.append(regression_weight)

                    missing_bias = nn.Parameter(torch.FloatTensor(layer_size))
                    nn.init.constant(tensor=missing_bias, val=0.0)
                    self.register_parameter(
                        name='regression_{!s}_{!s}_tau'.format(i+1, "t-{!s}".format(window+1)), param=missing_bias
                    )
                    missing.append(missing_bias)
                    self.regression_parameters.append(missing_bias)

                regression_bias = nn.Parameter(torch.FloatTensor(layer_size))
                nn.init.constant(tensor=regression_bias, val=0.0)
                self.register_parameter(name='regression_{!s}_bias'.format(i+1), param=regression_bias)
                self.regression_biases.append(regression_bias)
                self.regression_parameters.append(regression_bias)

                self.regression_weights.append(weights)
                self.missing_biases.append(missing)
            else:
                self.regression_weights.append(None)
                self.regression_biases.append(None)
                self.missing_biases.append(None)


    def forward(self, xs=None):
        """
        Builds the Temporal GSN computation graph, either from the input batch sequence `xs`
        or generating from the starting point of `hiddens`
        """
        # first go through the GSN to get the current hiddens, then regression to next hiddens
        # then generate new x from gsn
        sequence_hiddens = []
        sequence_xs = []
        sequence_x_samples = []
        if xs is not None:
            for x in xs:
                _, hiddens, _ = self.gsn.forward(x=x)
                sequence_hiddens.append(hiddens)
                next_hiddens = self.regression_step(sequence_hiddens)
                recon_xs, _, samples = self.gsn.forward(x=None, hiddens=next_hiddens)
                sequence_xs.append(recon_xs)
                sequence_x_samples.append(samples)

        return sequence_xs, sequence_hiddens, sequence_x_samples

    def generate(self, x=None, hiddens=None, n_samples=10):
        xs = []
        sequence_hiddens = []
        if x is not None:
            _, hiddens, _ = self.gsn.forward(x=x)
            xs.append(x)
        sequence_hiddens.append(hiddens)

        for i in range(n_samples):
            next_hiddens = self.regression_step(sequence_hiddens)
            recon_xs, hiddens, _ = self.gsn.forward(x=None, hiddens=next_hiddens)
            xs.append(recon_xs[-1])
            sequence_hiddens.append(hiddens)

        return xs

    def regression_step(self, sequence_hiddens):
        """
        Given the history list of GSN hiddens, make the next full list of gsn hiddens from our regression parameters
        """
        sequence_reverse = sequence_hiddens[::-1]
        hiddens = []
        for layer, _ in enumerate(self.sizes[1:]):
            if layer % 2 == 0:
                # do the window calculation for the layer!
                regression_terms = []
                for window in range(self.window_size):
                    if window < len(sequence_reverse):
                        regression_weight = self.regression_weights[layer][window]
                        regression_bias = self.regression_biases[layer] if window == 0 else None
                        regression_terms.append(
                            F.linear(
                                input=sequence_reverse[window][layer], weight=regression_weight, bias=regression_bias
                            )
                        )
                    else:
                        regression_terms.append(self.missing_biases[layer][window])
                hiddens.append(sum(regression_terms))
            else:
                hiddens.append(None)
        return hiddens


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        SequencedMNIST(
            '../datasets', sequence=2, length=21,
            transform=transforms.ToTensor(), sequence_transform=transforms.Lambda(lambda seq: torch.stack(seq))
        ),
        batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        SequencedMNIST(
            '../datasets', sequence=2, length=21,
            transform=transforms.ToTensor(), sequence_transform=transforms.Lambda(lambda seq: torch.stack(seq))
        ),
        batch_size=32, shuffle=True
    )

    model = TGSN(sizes=[784, 1500, 1500], window_size=2, tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.Tanh(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True)
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    gsn_optimizer = optim.Adam(model.gsn.parameters(), lr=.0003)
    regression_optimizer = optim.Adam(model.regression_parameters, lr=.0003)

    times = []
    epochs = 300
    for epoch in range(epochs):
        print("Epoch", epoch)
        epoch_start = time.time()
        #####
        # train the gsn
        #####
        model.train()
        gsn_train_losses = []
        gsn_start_time = time.time()
        for batch_idx, (sequence_batch, _) in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            flat_sequence_batch = sequence_batch.view(batch_size * sequence_len, rest)
            # break up this potentially large batch into nicer small ones for gsn
            for batch_idx in range(int(flat_sequence_batch.size()[0] / 64)):
                x = flat_sequence_batch[batch_idx*64:(batch_idx+1)*64]
                # train the gsn!
                gsn_optimizer.zero_grad()
                recons, _, _ = model.gsn(x)
                losses = [F.binary_cross_entropy(input=recon, target=x) for recon in recons]
                loss = sum(losses)
                loss.backward()
                gsn_optimizer.step()
                gsn_train_losses.append(losses[-1].data.numpy())

        print("GSN Train Loss", np.average(gsn_train_losses), "took {!s}".format(make_time_units_string(time.time()-gsn_start_time)))
        ####
        # train the regression step
        ####
        model.train()
        regression_train_losses = []
        for batch_idx, (sequence_batch, _) in enumerate(train_loader):
            _start = time.time()
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest)
            targets = sequence_batch[1:]

            regression_optimizer.zero_grad()
            predictions, _, _ = model(sequence_batch)
            losses = [
                [F.binary_cross_entropy(input=recon, target=targets[step]) for recon in recons]
                for step, recons in enumerate(predictions[:-1])
            ]
            loss = sum([sum(l) for l in losses])
            loss.backward()
            regression_optimizer.step()
            regression_train_losses.append(losses[-1][-1].data.numpy())

        print("Regression Train Loss", np.average(regression_train_losses))
        example, _ = train_loader.dataset[0]
        example = Variable(example, requires_grad=False)
        if use_cuda:
            example = example.cuda()
        sequence_len = example.size()[0]
        rest = int(np.prod(example.size()[1:]))
        flat_example = example.view(sequence_len, 1, rest)
        preds, _, _ = model(flat_example)
        preds = torch.stack([flat_example[0]] + [recons[-1] for recons in preds])
        preds = preds.view(sequence_len+1, 1, 28, 28)
        save_image(preds.data, '{!s}.png'.format(epoch), nrow=10)

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
