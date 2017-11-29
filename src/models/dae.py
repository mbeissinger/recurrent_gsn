"""
Denoising autoencoder
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


class DAE(nn.Module):
    def __init__(self, sizes, tied_weights=False, noise=.2, visible_act=nn.Sigmoid()):
        super().__init__()
        sizes_reverse = sizes[::-1]
        self.tied_weights = tied_weights
        self.visible_act = visible_act
        self.noise = noise
        self.encode_layers = []
        self.decode_layers = []
        self.decode_params = []
        i = 0
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(in_features=in_size, out_features=out_size, bias=True)
            self.add_module("layer_{!s}".format(i), layer)
            self.encode_layers.append(layer)
            i += 1
        if not tied_weights:
            for in_size, out_size in zip(sizes_reverse[:-1], sizes_reverse[1:]):
                layer = nn.Linear(in_features=in_size, out_features=out_size, bias=True)
                self.add_module("layer_{!s}".format(i), layer)
                self.decode_layers.append(layer)
                i += 1
        else:
            for layer in self.encode_layers[::-1]:
                bias = nn.Parameter(torch.zeros(layer.weight.size()[1]))
                self.register_parameter('layer_{!s}_bias'.format(i), bias)
                i += 1
                self.decode_params.append([layer, bias])

        if isinstance(self.visible_act, nn.Sigmoid):
            self.input_corrupt = SaltAndPepper(amount=self.noise)
        else:
            self.input_corrupt = GaussianNoise(mean=0, std=self.noise)

    def forward(self, x):
        # flatten input
        if len(x.size()) > 2:
            x = x.view(-1, int(np.prod(x.size()[1:])))
        # corrupt input
        x = self.input_corrupt(x)
        corrupted = x
        # encode
        for layer in self.encode_layers:
            x = layer(x)
            x = F.relu(x)
        # decode
        if self.tied_weights:
            for i, (layer, bias) in enumerate(self.decode_params):
                x = F.linear(x, weight=layer.weight.t(), bias=bias)
                if i == len(self.decode_params)-1:
                    x = self.visible_act(x)
                else:
                    x = F.relu(x)
        else:
            for i, layer in enumerate(self.decode_layers):
                x = layer(x)
                if i == len(self.decode_layers)-1:
                    x = self.visible_act(x)
                else:
                    x = F.relu(x)

        return x, corrupted


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST('../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=32, shuffle=True
    )

    model = DAE(sizes=[784, 1024], noise=0.5, tied_weights=True)
    if use_cuda:
        model.cuda()
    print('Model:', model)
    optimizer = optim.Adam(model.parameters(), lr=.0003)

    for epoch in range(200):
        print("Epoch", epoch)
        model.train()
        train_losses = []
        for batch_idx, (image, _) in enumerate(train_loader):
            image = Variable(image, requires_grad=False)
            if use_cuda:
                image = image.cuda()
            optimizer.zero_grad()
            flat_image = image.view(-1, int(np.prod(image.size()[1:])))
            recon, _ = model(flat_image)
            loss = F.binary_cross_entropy(input=recon, target=flat_image)
            loss.backward()
            optimizer.step()
            train_losses.extend(loss.data)
        print("Train Loss", np.average(train_losses))
        example, _ = train_loader.dataset[0]
        example = Variable(example, requires_grad=False)
        if use_cuda:
            example = example.cuda()
        flat_example = example.view(1, 784)
        example_recon, example_noisy = model(flat_example)
        im = transforms.ToPILImage()(flat_example.view(1,28,28).cpu().data)
        im.save('{!s}_image.jpg'.format(epoch))
        noisy = transforms.ToPILImage()(example_noisy.view(1, 28, 28).cpu().data)
        noisy.save('{!s}_noisy.jpg'.format(epoch))
        r_im = transforms.ToPILImage()(example_recon.view(1,28,28).cpu().data)
        r_im.save('{!s}_recon.jpg'.format(epoch))

        model.eval()
        test_losses = []
        for batch_idx, (image, _) in enumerate(test_loader):
            image = Variable(image, volatile=True, requires_grad=False)
            if use_cuda:
                image = image.cuda()
            flat_image = image.view(-1, int(np.prod(image.size()[1:])))
            recon, _ = model(flat_image)
            test_loss = F.binary_cross_entropy(input=recon, target=flat_image)
            test_losses.extend(test_loss.data)
        print("Test Loss", np.average(test_losses))
