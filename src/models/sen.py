"""
@author: Markus Beissinger
University of Pennsylvania, 2017

This class produces the Sequence Encoder model discussed in the paper: (thesis link)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.models.gsn import GSN, act_to_string

use_cuda = torch.cuda.is_available()


class SEN(nn.Module):
    def __init__(self, in_size, hidden_size, tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True, rnn_hidden_size=256):
        super().__init__()
        """
        For now, simple 4-layer SEN, stacking GSN->LSTM->GSN->LSTM, way too many things hardcoded in this class right now XD
        There could be many other architectures, such as deconv autoencoder -> rnn -> denoising autoencoder -> rnn...
        etc.
        """
        self.hidden_act = hidden_act
        self.rnn_hidden_size = rnn_hidden_size
        # GSN1
        self.gsn_0 = GSN(
            sizes=[in_size, hidden_size, hidden_size], tied_weights=tied_weights, walkbacks=walkbacks, visible_act=visible_act, hidden_act=hidden_act,
            input_noise=input_noise, hidden_noise=hidden_noise, input_sampling=input_sampling, noiseless_h1=noiseless_h1
        )

        # lstm1
        self.lstm_0 = nn.LSTMCell(input_size=hidden_size, hidden_size=rnn_hidden_size)
        self.linear_0 = nn.Linear(in_features=rnn_hidden_size, out_features=hidden_size, bias=True)
        nn.init.xavier_uniform(
            tensor=self.lstm_0.weight_ih, gain=nn.init.calculate_gain('tanh')
        )
        nn.init.xavier_uniform(
            tensor=self.lstm_0.weight_hh, gain=nn.init.calculate_gain('tanh')
        )
        nn.init.constant(tensor=self.lstm_0.bias_ih, val=0.0)
        nn.init.constant(tensor=self.lstm_0.bias_hh, val=0.0)

        nn.init.xavier_uniform(
            tensor=self.linear_0.weight, gain=nn.init.calculate_gain(act_to_string(self.hidden_act))
        )
        nn.init.constant(tensor=self.linear_0.bias, val=0.0)

        # gsn 2
        gsn_2_sizes = [rnn_hidden_size*2, int(rnn_hidden_size*2*1.5), int(rnn_hidden_size*2*1.5)]
        self.gsn_1 = GSN(
            sizes=gsn_2_sizes, tied_weights=tied_weights, walkbacks=4, visible_act=nn.Tanh(), hidden_act=hidden_act,
            input_noise=hidden_noise, hidden_noise=hidden_noise, input_sampling=False, noiseless_h1=noiseless_h1
        )
        # lstm 2
        self.lstm_1 = nn.LSTMCell(input_size=gsn_2_sizes[1], hidden_size=rnn_hidden_size)
        self.linear_1 = nn.Linear(in_features=rnn_hidden_size, out_features=gsn_2_sizes[1], bias=True)
        nn.init.xavier_uniform(
            tensor=self.lstm_1.weight_ih, gain=nn.init.calculate_gain('tanh')
        )
        nn.init.xavier_uniform(
            tensor=self.lstm_1.weight_hh, gain=nn.init.calculate_gain('tanh')
        )
        nn.init.constant(tensor=self.lstm_1.bias_ih, val=0.0)
        nn.init.constant(tensor=self.lstm_1.bias_hh, val=0.0)

        nn.init.xavier_uniform(
            tensor=self.linear_1.weight, gain=nn.init.calculate_gain(act_to_string(self.hidden_act))
        )
        nn.init.constant(tensor=self.linear_1.bias, val=0.0)

        self.input_criterion = F.binary_cross_entropy if isinstance(visible_act, nn.Sigmoid) else F.mse_loss

    def forward(self, xs):
        """
        Builds the SEN computation graph, from the input batch sequence `xs`
        """
        # first go through the GSN to get the current hiddens, then regression to next hiddens
        sequence_xs = []
        target_recons = []

        encode_cost = []
        regression_cost = []

        if xs is not None:
            lstm_0_hiddens = (None, None)
            lstm_1_hiddens = (None, None)
            for x in xs:
                # encode X->H_0
                x_recons, gsn_h_0, _ = self.gsn_0.forward(x=x)
                encode_cost.extend([self.input_criterion(input=recon, target=x) for recon in x_recons])

                # regression H_0->H_0, producing regression H_1
                h_0 = gsn_h_0[0]  # grab the gsn hidden layer
                lstm_h1, lstm_c1 = lstm_0_hiddens  # grab the lstm hiddens
                # init if they aren't there
                if lstm_h1 is None or lstm_c1 is None:
                    lstm_h1, lstm_c1 = self.init_rnn_hiddens(x=h_0, lstm_cell=self.lstm_0)
                # do the lstm
                lstm_h1, lstm_c1 = self.lstm_0(h_0, (lstm_h1, lstm_c1))

                # encode H_1->H_2
                h_1 = torch.cat([lstm_h1, lstm_c1], dim=1)
                h_1_recons, gsn_h_2, _ = self.gsn_1.forward(x=h_1)
                # encode_cost.extend([F.mse_loss(input=h_1_recon, target=h_1) for h_1_recon in h_1_recons])

                # regression H_2->H_2, producing H_3
                h_2 = gsn_h_2[0]  # grab the gsn hidden layer
                lstm_h3, lstm_c3 = lstm_1_hiddens  # grab the lstm hiddens
                # init if they aren't there
                if lstm_h3 is None or lstm_c3 is None:
                    lstm_h3, lstm_c3 = self.init_rnn_hiddens(x=h_2, lstm_cell=self.lstm_1)
                # do the lstm
                lstm_h3, lstm_c3 = self.lstm_1(h_2, (lstm_h3, lstm_c3))
                lstm_1_hiddens = (lstm_h3, lstm_c3)  # store these lstm vars for next iteration

                # decode rnn H_3->H_2
                next_h_2 = self.linear_1(lstm_h3)

                # use gsn to generate the next h_2 -> h_1
                next_h_1s, _, _ = self.gsn_1.forward(hiddens=[next_h_2, None])
                next_h_1 = next_h_1s[-1]

                # decode rnn h_1->h_0
                lstm_h1, lstm_c1 = torch.split(next_h_1, split_size=self.rnn_hidden_size, dim=1)
                lstm_0_hiddens = (lstm_h1, lstm_c1)
                next_h0 = self.linear_0(lstm_h1)

                # gsn generate H_0->X
                next_xs, _, _ = self.gsn_0.forward(hiddens=[next_h0, None])
                target_recons.append(next_xs)
                sequence_xs.append(next_xs[-1])

        return sequence_xs, sum(encode_cost), target_recons

    def init_rnn_hiddens(self, x, lstm_cell):
        batch_size = x.size()[0]
        return [
            Variable(
                torch.zeros(batch_size, lstm_cell.hidden_size), requires_grad=False).cuda() if x.is_cuda  # puts the tensor on gpu if our input is on gpu
            else Variable(torch.zeros(batch_size, lstm_cell.hidden_size), requires_grad=False)
            for _ in range(2)
        ]

    def generate(self, x=None, hiddens=None, n_samples=10):
        pass


class AutoEncoderSEN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        pass


class DeconvSEN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        pass