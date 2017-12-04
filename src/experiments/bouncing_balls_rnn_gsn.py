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
from src.data.bouncing_balls import BouncingBalls

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski'),
        batch_size=2, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski'),
        batch_size=2, shuffle=True
    )

    model = TGSN(sizes=[225, 1500, 1500], window_size=2, tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(), hidden_act=nn.Tanh(),
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
        for batch_idx, sequence_batch in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            flat_sequence_batch = sequence_batch.view(batch_size * sequence_len, rest)
            # break up this potentially large batch into nicer small ones for gsn
            for batch_idx in range(int(flat_sequence_batch.size()[0] / 32)):
                x = flat_sequence_batch[batch_idx*32:(batch_idx+1)*32]
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
        for batch_idx, sequence_batch in enumerate(train_loader):
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
            losses = [F.binary_cross_entropy(input=recons[-1], target=targets[step]) for step, recons in enumerate(predictions[:-1])]
            loss = sum(losses)
            loss.backward()
            regression_optimizer.step()
            regression_train_losses.append(losses[-1].data.numpy())

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
        preds = preds.view(sequence_len+1, 1, 15, 15)
        save_image(preds.data, '{!s}.png'.format(epoch), nrow=10)

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
