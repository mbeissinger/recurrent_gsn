import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.utils import make_time_units_string
from src.data.cmu_mocap import Mocap
from src.models.untied_gsn import UntiedGSN

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        Mocap(),
        batch_size=1, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        Mocap(mode='test'),
        batch_size=1, shuffle=True,
    )

    model = UntiedGSN(
        sizes=[49, 128, 128], visible_act=lambda x: x, hidden_act=nn.Tanh(),
        input_noise=0, hidden_noise=0, input_sampling=False, noiseless_h1=True
    )
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
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
            sequence_batch = Variable(sequence_batch, requires_grad=False).float()
            if use_cuda:
                sequence_batch = sequence_batch.cuda()
            sequence = sequence_batch.squeeze(dim=0)
            subsequences = torch.split(sequence, split_size=100)
            for seq in subsequences:
                batch_size = 1
                seq_len = seq.size()[0]
                seq = seq.view(seq_len, -1).contiguous()
                seq = seq.unsqueeze(dim=1)
                targets = seq[1:]

                optimizer.zero_grad()
                predictions = model(seq)
                losses = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), .25)
                optimizer.step()
                train_losses.append(np.mean([l.data.cpu().numpy() for l in losses]))

                accuracies = [F.mse_loss(input=pred, target=targets[step]) for step, pred in enumerate(predictions[:-1])]
                train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

                acc = []
                p = torch.cat(predictions[:-1]).view(batch_size, seq_len - 1, 49).contiguous()
                t = targets.view(batch_size, seq_len - 1, 49).contiguous()
                for i, px in enumerate(p):
                    tx = t[i]
                    acc.append(torch.sum((tx - px) ** 2) / len(px))
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
            sequence_batch = Variable(sequence_batch, requires_grad=False, volatile=True).float()
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
            p = torch.cat(predictions[:-1]).view(batch_size, sequence_len - 1, rest).contiguous()
            t = targets.view(batch_size, sequence_len - 1, rest).contiguous()
            for i, px in enumerate(p):
                tx = t[i]
                acc.append(torch.sum((tx - px) ** 2) / len(px))
            test_accuracies2.append(np.mean([a.data.cpu().numpy() for a in acc]))

        print("Test Accuracy", np.mean(test_accuracies))
        print("Test Accuracy2", np.mean(test_accuracies2))
        print("Test time", make_time_units_string(time.time() - _start))

        with open('_mocap_untied_train.csv', 'a') as f:
            lines = ['{!s},{!s},{!s}\n'.format(loss, acc, acc2) for loss, acc, acc2 in zip(train_losses, train_accuracies, train_accuracies2)]
            for line in lines:
                f.write(line)
        with open('_mocap_untied_.csv', 'a') as f:
            f.write('{!s},{!s},{!s},{!s},{!s}\n'.format(np.mean(train_losses), np.mean(train_accuracies), np.mean(test_accuracies), np.mean(train_accuracies2), np.mean(test_accuracies2)))

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
