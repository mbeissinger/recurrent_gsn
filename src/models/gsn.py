"""
This module gives an implementation of the Generative Stochastic Network model.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN
'Deep Generative Stochastic Networks Trainable by Backprop'
Yoshua Bengio, Eric Thibodeau-Laufer
http://arxiv.org/abs/1306.1091

Scheduled noise is discussed in the paper:
'Scheduled denoising autoencoders'
Krzysztof J. Geras, Charles Sutton
http://arxiv.org/abs/1406.3269

TODO:
Multimodal transition operator (using NADE) discussed in:
'Multimodal Transitions for Generative Stochastic Networks'
Sherjil Ozair, Li Yao, Yoshua Bengio
http://arxiv.org/abs/1312.5578
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST

from src.models.noise import SaltAndPepper, GaussianNoise

use_cuda = torch.cuda.is_available()


def act_to_string(act):
    if isinstance(act, nn.Sigmoid):
        return 'sigmoid'
    elif isinstance(act, nn.Tanh):
        return 'tanh'
    elif isinstance(act, nn.ReLU):
        return 'relu'
    elif isinstance(act, nn.LeakyReLU):
        return 'leaky_relu'
    else:
        return 'linear'


class GSN(nn.Module):
    def __init__(self, sizes, tied_weights=True, walkbacks=5, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=True, noiseless_h1=True):
        super().__init__()
        self.sizes = sizes
        self.visible_act = visible_act  # activation for visible layer - should be appropriate for input data type.
        self.hidden_act = hidden_act  # activation for hidden layers
        self.walkbacks = walkbacks  # number of walkbacks (generally 2*hidden layers) - need enough to have info from top layer propagate to visible layer
        self.input_sampling = input_sampling  # whether to sample at each walkback step - makes it like Gibbs sampling.
        self.noiseless_h1 = noiseless_h1  # whether to keep the first hidden layer uncorrupted
        # Noise to add to the visible and hidden layers - salt/pepper for binary, gaussian for real values
        if isinstance(self.visible_act, nn.Sigmoid):
            self.input_corrupt = SaltAndPepper(amount=input_noise)
        else:
            self.input_corrupt = GaussianNoise(mean=0, std=input_noise)
        self.hidden_corrupt = GaussianNoise(mean=0, std=hidden_noise)

        # create the parameters and use F.linear to make things simpler doing tied weights and stuff
        self.layers = []  # [(encode weight, bias), (decode weight, bias)]
        for i, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            #### encode part
            ### linear layer weight encode from in to out
            encode_weight = nn.Parameter(torch.FloatTensor(output_size, input_size))  # (outs, ins) - see nn.Linear class
            # initialize our weight (make sure the gain is calculated for the resulting nonlinearity
            nn.init.xavier_uniform(tensor=encode_weight, gain=nn.init.calculate_gain(act_to_string(self.hidden_act)))
            self.register_parameter(name='layer_{!s}_{!s}_encode_weight'.format(i, i+1), param=encode_weight)
            ### linear layer bias encode in to out
            encode_bias = nn.Parameter(torch.FloatTensor(output_size))
            nn.init.constant(tensor=encode_bias, val=0.0)
            self.register_parameter(name='layer_{!s}_{!s}_encode_bias'.format(i, i+1), param=encode_bias)
            #### decode part
            if tied_weights:
                decode_weight = encode_weight.t()
            else:
                decode_weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
                nn.init.xavier_uniform(tensor=decode_weight,
                                       gain=nn.init.calculate_gain(act_to_string(self.visible_act if i == 0 else self.hidden_act)))
                self.register_parameter(name='layer_{!s}_{!s}_decode_weight'.format(i+1, i), param=decode_weight)
            decode_bias = nn.Parameter(torch.FloatTensor(input_size))
            nn.init.constant(tensor=decode_bias, val=0.0)
            self.register_parameter(name='layer_{!s}_{!s}_decode_bias'.format(i+1, i), param=decode_bias)

            self.layers.append(((encode_weight, encode_bias), (decode_weight, decode_bias)))

    def forward(self, x=None, hiddens=None):
        """
        Builds the GSN computation graph, either from the input `x` or generating from the starting point of `hiddens`
        """
        xs = []
        if hiddens is None and x is not None:
            # if we are starting from x, initialize the hidden activations to be 0!
            batch_size = x.size()[0]
            hiddens = [
                Variable(torch.zeros(batch_size, h_size)).cuda() if x.is_cuda  # puts the tensor on gpu if our input is on gpu
                else Variable(torch.zeros(batch_size, h_size))
                for h_size in self.sizes[1:]
            ]
        if x is not None:
            # run the normal computation graph from input x
            x_recon = x
            for _ in range(self.walkbacks):
                hiddens = self.encode(x_recon, hiddens)
                x_recon, hiddens = self.decode(hiddens)
                xs.append(x_recon)
        else:
            # run the generative computation graph from hiddens H
            for _ in range(self.walkbacks):
                x_recon, hiddens = self.decode(hiddens)
                hiddens = self.encode(x_recon, hiddens)
                xs.append(x_recon)

        return xs, hiddens

    def encode(self, x, hiddens):
        """
        Given the value for x and hidden activations, do an encoding pass (update every other hidden activation)
        """
        # starting with x and the hiddens,
        # update every other hidden layer using the activations from layers below and above
        corrupted_x = self.input_corrupt(x)
        for i in range(0, len(hiddens), 2):  # even layers
            # grab the parameters to use!
            (encode_w, encode_b), _ = self.layers[i]
            # encode up from below
            # if first layer, use x, otherwise use the hidden from below
            if i == 0:
                below = corrupted_x
            else:
                below = hiddens[i-1]
            hidden = F.linear(input=below, weight=encode_w, bias=encode_b)

            # decode down from above (if this isn't the top layer)
            if i < len(hiddens)-1:
                _, (decode_w, decode_b) = self.layers[i+1]
                hidden = hidden + F.linear(input=hiddens[i+1], weight=decode_w, bias=decode_b)

            # pre-activation noise
            if not (i == 0 and self.noiseless_h1):
                hidden = self.hidden_corrupt(hidden)

            # apply activation
            hidden = self.hidden_act(hidden)

            # post-activation noise
            if not (i == 0 and self.noiseless_h1):
                hidden = self.hidden_corrupt(hidden)

            # donezo for the hidden layer
            hiddens[i] = hidden

        return hiddens

    def decode(self, hiddens):
        """
        Given the value for the hidden activations, do a decoding pass
        (update every other hidden activation and the visible input layer
        """
        # starting with the hiddens,
        # update the reconstructed x and every other hidden layer using the activations from layers below and above
        for i in range(1, len(hiddens), 2):  # odd layers
            # grab the parameters to use!
            (encode_w, encode_b), _ = self.layers[i]
            # encode up from below
            hidden = F.linear(input=hiddens[i-1], weight=encode_w, bias=encode_b)

            # decode down from above (if this isn't the top layer)
            if i < len(hiddens) - 1:
                _, (decode_w, decode_b) = self.layers[i+1]
                hidden = hidden + F.linear(input=hiddens[i+1], weight=decode_w, bias=decode_b)

            # pre-activation noise
            hidden = self.hidden_corrupt(hidden)

            # apply activation
            hidden = self.hidden_act(hidden)

            # post-activation noise
            hidden = self.hidden_corrupt(hidden)

            # donezo for the hidden layer
            hiddens[i] = hidden

        # now do the reconstructed x!
        _, (decode_w, decode_b) = self.layers[0]
        x_recon = F.linear(input=hiddens[0], weight=decode_w, bias=decode_b)
        x_recon = self.visible_act(x_recon)
        # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL
        if self.input_sampling:
            if isinstance(self.visible_act, nn.Sigmoid):
                x_recon = torch.bernoulli(x_recon)
            else:
                print("Input sampling isn't defined for activation {!s}".format(type(self.visible_act)))

        return x_recon, hiddens


if __name__ == '__main__':
    def binarize(x, thresh=0.5):
        x[x<=thresh] = 0
        x[x>thresh] = 1
        return x

    train_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: binarize(x))
                       ])),
        batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: binarize(x))
        ])),
        batch_size=32, shuffle=True
    )

    model = GSN(sizes=[784, 1024, 1024], tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
                 input_noise=.4, hidden_noise=2, input_sampling=False, noiseless_h1=True)
    if use_cuda:
        model.cuda()
    print('Model:', model)
    optimizer = optim.Adam(model.parameters(), lr=.0003)

    for epoch in range(200):
        print("Epoch", epoch)
        model.train()
        train_losses = []
        for batch_idx, (image_batch, _) in enumerate(train_loader):
            image_batch = Variable(image_batch, requires_grad=False)
            if use_cuda:
                image_batch = image_batch.cuda()
            optimizer.zero_grad()
            flat_image_batch = image_batch.view(-1, int(np.prod(image_batch.size()[1:])))
            recons, _ = model(flat_image_batch)
            loss = 0.
            for recon in recons:
                loss += F.binary_cross_entropy(input=recon, target=flat_image_batch)
            loss = loss / len(recons)
            loss.backward()
            optimizer.step()
            train_losses.extend(loss.data)
        print("Train Loss", np.average(train_losses))
        example, _ = train_loader.dataset[0]
        example = Variable(example, requires_grad=False)
        if use_cuda:
            example = example.cuda()
        flat_example = example.view(1, 784)
        example_recons, _ = model(flat_example)
        example_recon = example_recons[-1]
        im = transforms.ToPILImage()(flat_example.view(1,28,28).cpu().data)
        im.save('{!s}_image.png'.format(epoch))
        r_im = transforms.ToPILImage()(example_recon.view(1,28,28).cpu().data)
        r_im.save('{!s}_recon.png'.format(epoch))

        model.eval()
        test_losses = []
        for batch_idx, (image_batch, _) in enumerate(test_loader):
            image_batch = Variable(image_batch, volatile=True, requires_grad=False)
            if use_cuda:
                image_batch = image_batch.cuda()
            flat_image_batch = image_batch.view(-1, int(np.prod(image_batch.size()[1:])))
            recons, _ = model(flat_image_batch)
            test_loss = 0.
            for recon in recons:
                test_loss += F.binary_cross_entropy(input=recon, target=flat_image_batch)
            test_loss = test_loss / len(recons)
            test_losses.extend(test_loss.data)
        print("Test Loss", np.average(test_losses))
