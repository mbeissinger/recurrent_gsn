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
from src.models.lstm import LSTM

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski'),
        batch_size=4, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski', mode='test'),
        batch_size=4, shuffle=True,
    )
    example = test_loader.dataset[0]
    example = Variable(torch.Tensor(example), requires_grad=False)
    if use_cuda:
        example = example.cuda()
    sequence_len = example.size()[0]
    rest = int(np.prod(example.size()[1:]))
    flat_example = example.view(sequence_len, 1, rest)
    save_image(flat_example.view(sequence_len, 1, 15, 15).data.cpu(), '_bouncing_balls_lstm_real_example.png', nrow=10)
    images = [ToPILImage()(img) for img in flat_example.view(sequence_len, 1, 15, 15).data.cpu()]
    with open('_bouncing_balls_lstm_real_example.gif', 'wb') as f:
        images[0].save(f, save_all=True, append_images=images[1:])

    model = LSTM(
        input_size=15*15, hidden_size=1500,
        num_layers=1, bias=True, batch_first=False,
        dropout=0, bidirectional=False, output_size=15*15, output_activation=nn.Sigmoid()
    )
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    # optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = optim.Adam(model.parameters(), lr=3e-2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    times = []
    epochs = 500
    for epoch in range(epochs):
        print("Epoch", epoch)
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_accuracies = []
        train_accuracies2 = []
        _start = time.time()
        for batch_idx, sequence_batch in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest).contiguous()
            targets = sequence_batch[1:]

            optimizer.zero_grad()
            predictions = model(sequence_batch)
            losses = [F.binary_cross_entropy(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            loss = sum(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), .25)
            optimizer.step()
            train_losses.append(np.mean([l.data.cpu().numpy() for l in losses]))

            accuracies = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))
            acc = []
            p = predictions[:-1].view(batch_size, sequence_len-1, rest).contiguous()
            t = targets.view(batch_size, sequence_len-1, rest).contiguous()
            for i, px in enumerate(p):
                tx = t[i]
                acc.append(torch.sum((tx - px)**2)/len(px))
            train_accuracies2.append(np.mean([a.data.cpu().numpy() for a in acc]))

        print("Train Loss", np.mean(train_losses))
        print("Train Accuracy", np.mean(train_accuracies))
        print("Train Accuracy2", np.mean(train_accuracies2))
        print("Train time", make_time_units_string(time.time()-_start))

        model.eval()
        test_accuracies = []
        test_accuracies2 = []
        _start = time.time()
        for batch_idx, sequence_batch in enumerate(test_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False, volatile=True)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            batch_size = sequence_batch.size()[0]
            sequence_len = sequence_batch.size()[1]
            rest = int(np.prod(sequence_batch.size()[2:]))
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest).contiguous()
            targets = sequence_batch[1:]

            predictions = model(sequence_batch)
            accuracies = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
            test_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

            acc = []
            p = predictions[:-1].view(batch_size, sequence_len - 1, rest).contiguous()
            t = targets.view(batch_size, sequence_len - 1, rest).contiguous()
            for i, px in enumerate(p):
                tx = t[i]
                acc.append(torch.sum((tx - px) ** 2) / len(px))
            test_accuracies2.append(np.mean([a.data.cpu().numpy() for a in acc]))

        print("Test Accuracy", np.mean(test_accuracies))
        print("Test Accuracy2", np.mean(test_accuracies2))
        print("Test time", make_time_units_string(time.time() - _start))

        preds = model(flat_example)
        preds = torch.cat([torch.unsqueeze(flat_example[0], 0), preds])
        preds = preds.view(sequence_len + 1, 1, 15, 15)
        save_image(preds.data.cpu(), '_bouncing_balls_lstm_{!s}.png'.format(epoch), nrow=10)
        images = [ToPILImage()(pred) for pred in preds.data.cpu()]
        with open('_bouncing_balls_lstm_{!s}.gif'.format(epoch), 'wb') as fp:
            images[0].save(fp, save_all=True, append_images=images[1:])

        with open('_bouncing_balls_lstm_train.csv', 'a') as f:
            lines = ['{!s},{!s},{!s}\n'.format(loss, acc, acc2) for loss, acc, acc2 in zip(train_losses, train_accuracies, train_accuracies2)]
            for line in lines:
                f.write(line)
        with open('_bouncing_balls_lstm.csv', 'a') as f:
            f.write('{!s},{!s},{!s},{!s},{!s}\n'.format(np.mean(train_losses), np.mean(train_accuracies), np.mean(test_accuracies), np.mean(train_accuracies2), np.mean(test_accuracies2)))

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
