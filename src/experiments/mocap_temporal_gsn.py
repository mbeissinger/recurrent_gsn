import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.utils import make_time_units_string
from src.data.cmu_mocap import Mocap
from src.models.temporal_gsn import TGSN

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

    model = TGSN(
        sizes=[49, 128, 128], window_size=3, tied_weights=True, walkbacks=4, visible_act=lambda x: x,
        hidden_act=nn.Tanh(),
        input_noise=.1, hidden_noise=.5, input_sampling=False, noiseless_h1=True
    )
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    gsn_optimizer = optim.Adam(model.gsn.parameters(), lr=1e-3)
    regression_optimizer = optim.Adam(model.regression_parameters, lr=1e-3)

    times = []
    epochs = 1000
    for epoch in range(epochs):
        print("Epoch", epoch)
        epoch_start = time.time()
        model.train()
        #####
        # train the gsn
        #####
        gsn_train_losses = []
        gsn_train_accuracies = []
        gsn_start_time = time.time()
        for batch_idx, sequence_batch in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False).float()
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
                regression_optimizer.zero_grad()
                recons, _, _ = model.gsn(x)
                losses = [F.mse_loss(input=recon, target=x) for recon in recons]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), .25)
                gsn_optimizer.step()
                gsn_train_losses.append(losses[-1].data.cpu().numpy()[0])
                accuracies = [F.mse_loss(input=recon, target=x) for recon in recons]
                gsn_train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

        print("GSN Train Loss", np.mean(gsn_train_losses))
        print("GSN Train Accuracy", np.mean(gsn_train_accuracies))
        print("GSN Train time", make_time_units_string(time.time() - gsn_start_time))

        ####
        # train the regression step
        ####
        regression_train_losses = []
        regression_train_accuracies = []
        regression_train_accuracies2 = []
        regression_start = time.time()
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

                regression_optimizer.zero_grad()
                gsn_optimizer.zero_grad()
                predictions, _, _ = model(seq)
                losses = [F.mse_loss(input=recons[-1], target=targets[step]) for step, recons in
                          enumerate(predictions[:-1])]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), .25)
                regression_optimizer.step()
                regression_train_losses.append(np.mean([l.data.cpu().numpy() for l in losses]))
                accuracies = [F.mse_loss(input=recons[-1], target=targets[step]) for step, recons in enumerate(predictions[:-1])]
                regression_train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

                acc = []
                p = [recons[-1] for recons in predictions[:-1]]
                p = torch.cat(p).view(batch_size, seq_len - 1, 49)
                t = targets.view(batch_size, seq_len - 1, 49)
                for i, px in enumerate(p):
                    tx = t[i]
                    acc.append(torch.sum((tx - px) ** 2) / len(px))
                regression_train_accuracies2.append(np.mean([a.data.cpu().numpy() for a in acc]))


        print("Regression Train Loss", np.mean(regression_train_losses))
        print("Regression Train Accuracy", np.mean(regression_train_accuracies))
        print("Regression Train Accuracy2", np.mean(regression_train_accuracies2))
        print("Regression Train time", make_time_units_string(time.time() - regression_start))

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
            sequence_batch = sequence_batch.view(sequence_len, batch_size, rest)
            targets = sequence_batch[1:]

            predictions, _, _ = model(sequence_batch)
            accuracies = [F.mse_loss(input=recons[-1], target=targets[step]) for step, recons in enumerate(predictions[:-1])]
            test_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

            acc = []
            p = [recons[-1] for recons in predictions[:-1]]
            p = torch.cat(p).view(batch_size, sequence_len - 1, rest)
            t = targets.view(batch_size, sequence_len - 1, rest)
            for i, px in enumerate(p):
                tx = t[i]
                acc.append(torch.sum((tx - px) ** 2) / len(px))
            test_accuracies2.append(np.mean([a.data.cpu().numpy() for a in acc]))

        print("Test Accuracy", np.mean(test_accuracies))
        print("Test Accuracy2", np.mean(test_accuracies2))
        print("Test time", make_time_units_string(time.time() - _start))

        with open('_mocap_tgsn_gsn_train.csv', 'a') as f:
            lines = ['{!s},{!s}\n'.format(loss, acc) for
                     loss, acc in zip(gsn_train_losses, gsn_train_accuracies)]
            for line in lines:
                f.write(line)
        with open('_mocap_tgsn_reg_train.csv', 'a') as f:
            lines = ['{!s},{!s},{!s}\n'.format(loss, acc, acc2) for
                     loss, acc, acc2 in zip(regression_train_losses, regression_train_accuracies, regression_train_accuracies2)]
            for line in lines:
                f.write(line)
        with open('_mocap_tgsn_.csv', 'a') as f:
            f.write('{!s},{!s},{!s},{!s},{!s}\n'.format(np.mean(regression_train_losses), np.mean(regression_train_accuracies), np.mean(test_accuracies), np.mean(regression_train_accuracies2), np.mean(test_accuracies2)))

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
