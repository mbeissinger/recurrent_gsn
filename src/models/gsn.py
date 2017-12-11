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
from torchvision.utils import save_image

from src.models.noise import SaltAndPepper, GaussianNoise
from src.models.sampling import Binomial
from src.utils import make_time_units_string

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
        self.sampling_fn = Binomial()
        self.noiseless_h1 = noiseless_h1  # whether to keep the first hidden layer uncorrupted
        # Noise to add to the visible and hidden layers - salt/pepper for binary, gaussian for real values
        if isinstance(self.visible_act, nn.Sigmoid):
            self.input_corrupt = SaltAndPepper(amount=input_noise)
        else:
            self.input_corrupt = GaussianNoise(mean=0, std=input_noise)
        self.hidden_corrupt = GaussianNoise(mean=0, std=hidden_noise)

        # create the parameters and use F.linear to make things simpler doing tied weights and stuff
        self.layers = []  # [(encode weight, bias), decode weight]
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
            self.register_parameter(name='layer_{!s}_bias'.format(i+1), param=encode_bias)
            #### decode part
            if tied_weights:
                decode_weight = None
            else:
                decode_weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
                nn.init.xavier_uniform(tensor=decode_weight,
                                       gain=nn.init.calculate_gain(act_to_string(self.visible_act if i == 0 else self.hidden_act)))
                self.register_parameter(name='layer_{!s}_{!s}_decode_weight'.format(i+1, i), param=decode_weight)
            if i == 0:
                self.visible_bias = nn.Parameter(torch.FloatTensor(input_size))
                nn.init.constant(tensor=self.visible_bias, val=0.0)

            self.layers.append(((encode_weight, encode_bias), decode_weight))

    def forward(self, x=None, hiddens=None, requires_hidden_grad=False):
        """
        Builds the GSN computation graph, either from the input `x` or generating from the starting point of `hiddens`
        """
        xs = []  # reconstructed xs
        sampled_xs = []  # accumulate our sampled x from p(x|h) to use as the next input
        if hiddens is None and x is not None:
            # if we are starting from x, initialize the hidden activations to be 0!
            hiddens = self.init_hiddens(x, requires_grad=requires_hidden_grad)
        if x is not None:
            # run the normal computation graph from input x
            sampled_x = x
            for _ in range(self.walkbacks):
                hiddens = self.encode(sampled_x, hiddens)
                x_recon, hiddens, sampled_x = self.decode(hiddens)
                xs.append(x_recon)
                sampled_xs.append(sampled_x)
        else:
            # run the generative computation graph from hiddens H
            for _ in range(self.walkbacks):
                x_recon, hiddens, sampled_x = self.decode(hiddens)
                hiddens = self.encode(sampled_x, hiddens)
                xs.append(x_recon)
                sampled_xs.append(sampled_x)

        return xs, hiddens, sampled_xs

    def init_hiddens(self, x, requires_grad=False):
        batch_size = x.size()[0]
        return [
            Variable(torch.zeros(batch_size, h_size), requires_grad=requires_grad).cuda() if x.is_cuda  # puts the tensor on gpu if our input is on gpu
            else Variable(torch.zeros(batch_size, h_size), requires_grad=requires_grad)
            for h_size in self.sizes[1:]
        ]

    def encode(self, x, hiddens):
        """
        Given the value for x and hidden activations, do an encoding pass (update every other hidden activation)
        """
        # starting with x and the hiddens,
        # update every other hidden layer using the activations from layers below and above
        corrupted_x = self.input_corrupt(x)
        for i in range(0, len(hiddens), 2):  # even layers
            # grab the parameters to use!
            (encode_w, bias), _ = self.layers[i]
            # encode up from below
            # if first layer, use x, otherwise use the hidden from below
            if i == 0:
                below = corrupted_x
            else:
                below = hiddens[i-1]
            hidden = F.linear(input=below, weight=encode_w, bias=bias)

            # decode down from above (if this isn't the top layer)
            if i < len(hiddens)-1:
                (encode_w1, _), decode_w = self.layers[i+1]
                if decode_w is None:
                    decode_w = encode_w1.t()
                hidden = hidden + F.linear(input=hiddens[i+1], weight=decode_w)


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
            (encode_w, bias), _ = self.layers[i]
            # encode up from below
            hidden = F.linear(input=hiddens[i-1], weight=encode_w, bias=bias)

            # decode down from above (if this isn't the top layer)
            if i < len(hiddens) - 1:
                (encode_w1, _), decode_w = self.layers[i+1]
                if decode_w is None:
                    decode_w = encode_w1.t()
                hidden = hidden + F.linear(input=hiddens[i+1], weight=decode_w)

            # pre-activation noise
            hidden = self.hidden_corrupt(hidden)

            # apply activation
            hidden = self.hidden_act(hidden)

            # post-activation noise
            hidden = self.hidden_corrupt(hidden)

            # donezo for the hidden layer
            hiddens[i] = hidden

        # now do the reconstructed x!
        (encode_w1, _), decode_w = self.layers[0]
        if decode_w is None:
            decode_w = encode_w1.t()
        x_recon = F.linear(input=hiddens[0], weight=decode_w, bias=self.visible_bias)
        x_recon = self.visible_act(x_recon)
        # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL
        if self.input_sampling:
            if isinstance(self.visible_act, nn.Sigmoid):
                sampled = self.sampling_fn(x_recon)
            else:
                print("Input sampling isn't defined for activation {!s}".format(type(self.visible_act)))
                sampled = x_recon
        else:
            sampled = x_recon

        return x_recon, hiddens, sampled

    def generate_samples(self, x=None, hiddens=None, n=399):
        generated = []
        if hiddens is None and x is not None:
            hiddens = self.init_hiddens(x)
        if x is not None:
            sampled_x = x
            for _ in range(n):
                hiddens = self.encode(sampled_x, hiddens)
                x_recon, hiddens, sampled_x = self.decode(hiddens)
                generated.append(x_recon)
        else:
            for _ in range(n):
                x_recon, hiddens, sampled_x = self.decode(hiddens)
                hiddens = self.encode(sampled_x, hiddens)
                generated.append(x_recon)

        return generated


if __name__ == '__main__':
    import time
    def binarize(x, thresh=0.5):
        return (x>thresh).float()

    train_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=100, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=True
    )

    model = GSN(sizes=[784, 1000, 1000], tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.Tanh(),
                 input_noise=.2, hidden_noise=1, input_sampling=True, noiseless_h1=True)
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    optimizer = optim.Adam(model.parameters(), lr=.0003)

    times = []
    epochs = 300
    for epoch in range(epochs):
        print("Epoch", epoch)
        model.train()
        train_losses = []
        epoch_start = time.time()
        for batch_idx, (image_batch, _) in enumerate(train_loader):
            image_batch = Variable(image_batch, requires_grad=False)
            if use_cuda:
                image_batch = image_batch.cuda()
            optimizer.zero_grad()
            flat_image_batch = image_batch.view(-1, int(np.prod(image_batch.size()[1:])))
            recons, _, _ = model(flat_image_batch)
            losses = [F.binary_cross_entropy(input=recon, target=flat_image_batch) for recon in recons]
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            train_losses.append(losses[-1].data.numpy())
        print("Train Loss", np.average(train_losses))
        example, _ = train_loader.dataset[0]
        example = Variable(example, requires_grad=False)
        if use_cuda:
            example = example.cuda()
        flat_example = example.view(1, 784)
        example_recons, _, _ = model(flat_example)
        example_recon = example_recons[-1]
        im = transforms.ToPILImage()(flat_example.view(1,28,28).cpu().data)
        im.save('{!s}_image.png'.format(epoch))
        r_im = transforms.ToPILImage()(example_recon.view(1,28,28).cpu().data)
        r_im.save('{!s}_recon.png'.format(epoch))
        gen = 400
        n_side = int(np.sqrt(400))
        samples = model.generate_samples(x=flat_example, n=gen-1)
        samples = torch.stack([flat_example] + samples)
        samples = samples.view(gen, 1, 28, 28)
        save_image(samples.data, filename='{!s}_samples.png'.format(epoch), nrow=n_side)

        model.eval()
        test_losses = []
        for batch_idx, (image_batch, _) in enumerate(test_loader):
            image_batch = Variable(image_batch, volatile=True, requires_grad=False)
            if use_cuda:
                image_batch = image_batch.cuda()
            flat_image_batch = image_batch.view(-1, int(np.prod(image_batch.size()[1:])))
            recons, _, _ = model(flat_image_batch)
            test_loss = F.binary_cross_entropy(input=recons[-1], target=flat_image_batch)
            test_losses.append(test_loss.data.numpy())
        print("Test Loss", np.average(test_losses))
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
