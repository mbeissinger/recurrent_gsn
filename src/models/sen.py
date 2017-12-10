"""
@author: Markus Beissinger
University of Pennsylvania, 2017

This class produces the Sequence Encoder model discussed in the paper: (thesis link)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from src.models.gsn import GSN, act_to_string
from src.utils import make_time_units_string
from src.data.mnist_sequence import SequencedMNIST

use_cuda = torch.cuda.is_available()


class SEN(nn.Module):
    def __init__(self, sizes, tied_weights=True, walkbacks=5, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True, rnn_hidden_size=500):
        super().__init__()
        """
        For now, simple 4-layer SEN, stacking GSN->LSTM->GSN->LSTM
        There could be many other architectures, such as deconv autoencoder -> rnn -> denoising autoencoder -> rnn...
        etc.
        """
        self.sizes = sizes
        self.hidden_act = hidden_act
        # GSN1
        self.gsn_0 = GSN(
            sizes=sizes, tied_weights=tied_weights, walkbacks=walkbacks, visible_act=visible_act, hidden_act=hidden_act,
            input_noise=input_noise, hidden_noise=hidden_noise, input_sampling=input_sampling, noiseless_h1=noiseless_h1
        )
        # lstm1
        self.lstms_1 = [
            (
                nn.LSTMCell(input_size=size, hidden_size=rnn_hidden_size),
                nn.Linear(in_features=rnn_hidden_size, out_features=size, bias=True)
            ) for i, size in enumerate(sizes) if i % 2 == 1
        ]
        for i, (lstm, linear) in enumerate(self.lstms_1):
            self.add_module(name='lstm_cell_1_{!s}'.format(i), module=lstm)
            self.add_module(name='lstm_out_1_{!s}'.format(i), module=linear)
            nn.init.xavier_uniform(
                tensor=lstm.weight_ih, gain=nn.init.calculate_gain('tanh')
            )
            nn.init.xavier_uniform(
                tensor=lstm.weight_hh, gain=nn.init.calculate_gain('tanh')
            )
            if lstm.bias:
                nn.init.constant(tensor=lstm.bias_ih, val=0.0)
                nn.init.constant(tensor=lstm.bias_hh, val=0.0)

            nn.init.xavier_uniform(
                tensor=linear.weight, gain=nn.init.calculate_gain(act_to_string(self.hidden_act))
            )
            nn.init.constant(tensor=linear.bias, val=0.0)
        # gsn 2
        gsn_2_sizes = [rnn_hidden_size*2, int(rnn_hidden_size*2*1.5), int(rnn_hidden_size*2*1.5)]
        self.gsn_2 = GSN(
            sizes=gsn_2_sizes, tied_weights=tied_weights, walkbacks=4, visible_act=nn.Tanh(), hidden_act=hidden_act,
            input_noise=hidden_noise, hidden_noise=hidden_noise, input_sampling=False, noiseless_h1=noiseless_h1
        )
        # lstm 3
        self.lstms_3 = [
            (
                nn.LSTMCell(input_size=size, hidden_size=rnn_hidden_size),
                nn.Linear(in_features=rnn_hidden_size, out_features=size, bias=True)
            ) for i, size in enumerate(gsn_2_sizes) if i % 2 == 1
        ]
        for i, (lstm, linear) in enumerate(self.lstms_3):
            self.add_module(name='lstm_cell_3_{!s}'.format(i), module=lstm)
            self.add_module(name='lstm_out_3_{!s}'.format(i), module=linear)
            nn.init.xavier_uniform(
                tensor=lstm.weight_ih, gain=nn.init.calculate_gain('tanh')
            )
            nn.init.xavier_uniform(
                tensor=lstm.weight_hh, gain=nn.init.calculate_gain('tanh')
            )
            if lstm.bias:
                nn.init.constant(tensor=lstm.bias_ih, val=0.0)
                nn.init.constant(tensor=lstm.bias_hh, val=0.0)

            nn.init.xavier_uniform(
                tensor=linear.weight, gain=nn.init.calculate_gain(act_to_string(self.hidden_act))
            )
            nn.init.constant(tensor=linear.bias, val=0.0)

    def forward(self, xs):
        """
        Builds the SEN computation graph, from the input batch sequence `xs`
        """
        # first go through the GSN to get the current hiddens, then regression to next hiddens
        sequence_xs = []

        encode_cost = []
        regression_cost = []

        if xs is not None:
            lstm_hiddens = [[(None, None) for _ in self.lstms_1], [(None, None) for _ in self.lstms_3]]
            for x in xs:
                # encode X->H_0
                x_recons, h_0, _ = self.gsn_0.forward(x=x)

                # regression H_0->H_0
                lstm_1_hiddens = lstm_hiddens[0]
                for i, (lstm, linear) in enumerate(self.lstms_1):
                    h = h_0[i*2]  # grab the gsn hidden layer
                    lstm_ht, lstm_ct = lstm_1_hiddens[i]  # grab the lstm hiddens
                    # init if they aren't there
                    if lstm_ht is None or lstm_ct is None:
                        lstm_ht, lstm_ct = self.init_rnn_hiddens(x=h, lstm_cell=lstm)
                    # do the lstm
                    lstm_ht, lstm_ct = lstm(h, (lstm_ht, lstm_ct))
                    lstm_hiddens[0][i] = (lstm_ht, lstm_ct)  # store these lstm vars for next iteration

                # encode H_0->H_1
                for lstm_1_hiddens in lstm_hiddens[0]:
                    for h, c in lstm_1_hiddens:
                        pass

                # regression H_1->H_1

                # decode H_1->H_0

                # decode H_0->X







            h_t, c_t = None, None
            batch_size = 0
            for x in xs:
                if h_t is None or c_t is None:
                    h_t, c_t = self.init_rnn_hiddens(x)
                    batch_size = x.size()[0]
                _, hiddens, _ = self.gsn.forward(x=x)
                sequence_hiddens.append(hiddens)
                rnn_in = torch.cat([h for i, h in enumerate(hiddens) if i % 2 == 0])
                h_t, c_t = self.lstm_cell(rnn_in, (h_t, c_t))
                next_hiddens = self.lstm_out(h_t)
                processed_hiddens = []
                for i in range(len(hiddens)):
                    if i % 2 == 0:
                        processed_hiddens.append(next_hiddens[int(i/2*batch_size):int((i/2+1)*batch_size), :])
                    else:
                        processed_hiddens.append(None)
                recon_xs, _, samples = self.gsn.forward(x=None, hiddens=processed_hiddens)
                sequence_xs.append(recon_xs)
                sequence_x_samples.append(samples)

        return sequence_xs, sequence_hiddens, sequence_x_samples

    def init_rnn_hiddens(self, x, lstm_cell):
        batch_size = x.size()[0]
        return [
            Variable(
                torch.zeros(batch_size, lstm_cell.hidden_size), requires_grad=False).cuda() if x.is_cuda  # puts the tensor on gpu if our input is on gpu
            else Variable(torch.zeros(batch_size, lstm_cell.hidden_size), requires_grad=False)
            for _ in range(2)
        ]

    def generate(self, x=None, hiddens=None, n_samples=10):
        xs = []
        h_t, c_t = self.init_rnn_hiddens(x if x is not None else hiddens[0])
        if x is not None:
            _, hiddens, _ = self.gsn.forward(x=x)
            xs.append(x)

        for i in range(n_samples):
            rnn_in = torch.cat([h for i, h in enumerate(hiddens) if i % 2 == 0])
            h_t, c_t = self.lstm_cell(rnn_in, (h_t, c_t))
            next_hiddens = self.lstm_out(h_t)
            next_hiddens = [next_hiddens[int(i / 2 * batch_size):int((i / 2 + 1) * batch_size), :] if i % 2 == 0 else None for i in
                            range(len(hiddens))]
            recon_xs, hiddens, _ = self.gsn.forward(x=None, hiddens=next_hiddens)
            xs.append(recon_xs[-1])

        return xs


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

    model = RNNGSN(sizes=[784, 1500, 1500], rnn_hidden_size=500, tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.Tanh(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True)
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    gsn_optimizer = optim.Adam(model.gsn.parameters(), lr=.0003)
    regression_optimizer = optim.Adam(list(model.lstm_cell.parameters())+list(model.lstm_out.parameters()), lr=.0003)

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
            for batch_idx in range(int(flat_sequence_batch.size()[0] / 32)):
                x = flat_sequence_batch[batch_idx * 32:(batch_idx + 1) * 32]
                # train the gsn!
                gsn_optimizer.zero_grad()
                recons, _, _ = model.gsn(x)
                losses = [F.binary_cross_entropy(input=recon, target=x) for recon in recons]
                loss = sum(losses)
                loss.backward()
                gsn_optimizer.step()
                gsn_train_losses.append(losses[-1].data.cpu().numpy())

        print("GSN Train Loss", np.average(gsn_train_losses),
              "took {!s}".format(make_time_units_string(time.time() - gsn_start_time)))
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
            losses = [F.binary_cross_entropy(input=recons[-1], target=targets[step]) for step, recons in
                      enumerate(predictions[:-1])]
            loss = sum(losses)
            loss.backward()
            regression_optimizer.step()
            regression_train_losses.append(losses[-1].data.cpu().numpy())

        print("Regression Train Loss", np.average(regression_train_losses))
        example = train_loader.dataset[0]
        example = Variable(torch.Tensor(example), requires_grad=False)
        if use_cuda:
            example = example.cuda()
        sequence_len = example.size()[0]
        rest = int(np.prod(example.size()[1:]))
        flat_example = example.view(sequence_len, 1, rest)
        preds, _, _ = model(flat_example)
        preds = torch.stack([flat_example[0]] + [recons[-1] for recons in preds])
        preds = preds.view(sequence_len + 1, 1, 15, 15)
        save_image(preds.data.cpu(), '{!s}.png'.format(epoch), nrow=10)

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))


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