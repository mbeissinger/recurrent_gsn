import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

from src.utils import make_time_units_string
from src.data.bouncing_balls import BouncingBalls
from src.models.untied_gsn import UntiedGSN

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski'),
        batch_size=16, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski', mode='test'),
        batch_size=16, shuffle=True,
    )
    example = test_loader.dataset[0]
    example = Variable(torch.Tensor(example), requires_grad=False)
    if use_cuda:
        example = example.cuda()
    sequence_len = example.size()[0]
    rest = int(np.prod(example.size()[1:]))
    flat_example = example.view(sequence_len, 1, rest)
    save_image(flat_example.view(sequence_len, 1, 15, 15).data.cpu(), '_bouncing_untied_real_example.png', nrow=10)
    images = [ToPILImage()(img) for img in flat_example.view(sequence_len, 1, 15, 15).data.cpu()]
    with open('_bouncing_untied_real_example.gif', 'wb') as f:
        images[0].save(f, save_all=True, append_images=images[1:])

    model = UntiedGSN(
        sizes=[15*15, 500, 500], visible_act=nn.Sigmoid(), hidden_act=nn.ReLU(),
        input_noise=0., hidden_noise=0., input_sampling=True, noiseless_h1=True
    )
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    optimizer = optim.Adam(model.parameters(), lr=.0003)

    times = []
    epochs = 500
    for epoch in range(epochs):
        print("Epoch", epoch)
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_accuracies = []
        _start = time.time()
        for batch_idx, sequence_batch in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest)
            targets = sequence_batch[1:]

            optimizer.zero_grad()
            predictions = model(sequence_batch)
            losses = [F.binary_cross_entropy(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            train_losses.append(np.mean([l.data.cpu().numpy() for l in losses]))

            accuracies = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

        print("Train Loss", np.mean(train_losses))
        print("Train Accuracy", np.mean(train_accuracies))
        print("Train time", make_time_units_string(time.time()-_start))

        model.eval()
        test_accuracies = []
        _start = time.time()
        for batch_idx, sequence_batch in enumerate(test_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest)
            targets = sequence_batch[1:]

            predictions = model(sequence_batch)
            accuracies = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            test_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

        print("Test Accuracy", np.mean(test_accuracies))
        print("Test time", make_time_units_string(time.time() - _start))

        preds = model(flat_example)
        preds = torch.stack([flat_example[0]] + preds)
        preds = preds.view(sequence_len + 1, 1, 15, 15)
        save_image(preds.data.cpu(), '_bouncing_untied_{!s}.png'.format(epoch), nrow=10)
        images = [ToPILImage()(pred) for pred in preds.data.cpu()]
        with open('_bouncing_untied_{!s}.gif'.format(epoch), 'wb') as fp:
            images[0].save(fp, save_all=True, append_images=images[1:])

        with open('_bouncing_untied_train.csv', 'a') as f:
            lines = ['{!s},{!s}\n'.format(loss, acc) for loss, acc in zip(train_losses, train_accuracies)]
            for line in lines:
                f.write(line)
        with open('_bouncing_untied_.csv', 'a') as f:
            f.write('{!s},{!s},{!s}\n'.format(np.mean(train_losses), np.mean(train_accuracies), np.mean(test_accuracies)))

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
