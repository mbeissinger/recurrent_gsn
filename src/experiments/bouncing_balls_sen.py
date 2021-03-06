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
from src.models.sen import SEN

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski', train_size=200),
        batch_size=1, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        BouncingBalls(paper='boulanger-lewandowski', mode='test', train_size=200),
        batch_size=1, shuffle=True,
    )
    example = test_loader.dataset[0]
    example = Variable(torch.Tensor(example), requires_grad=False)
    if use_cuda:
        example = example.cuda()
    sequence_len = example.size()[0]
    rest = int(np.prod(example.size()[1:]))
    flat_example = example.view(sequence_len, 1, rest)
    save_image(flat_example.view(sequence_len, 1, 15, 15).data.cpu(), '_bouncing_sen_real_example.png', nrow=10)
    images = [ToPILImage()(img) for img in flat_example.view(sequence_len, 1, 15, 15).data.cpu()]
    with open('_bouncing_sen_real_example.gif', 'wb') as f:
        images[0].save(f, save_all=True, append_images=images[1:])

    model = SEN(
        in_size=15*15, hidden_size=500, rnn_hidden_size=500, tied_weights=True, walkbacks=4, visible_act=nn.Sigmoid(),
        hidden_act=nn.Tanh(),
        input_noise=0, hidden_noise=0, input_sampling=True, noiseless_h1=True
    )
    if use_cuda:
        model.cuda()
    print('Model:', model)
    print('Params:', [name for name, p in model.state_dict().items()])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # gsn_optimizer = optim.Adam(model.gsn.parameters(), lr=1e-3)
    # regression_optimizer = optim.Adam(list(model.lstm_cell.parameters())+list(model.lstm_out.parameters()), lr=1e-3)

    times = []
    epochs = 500
    for epoch in range(epochs):
        print("Epoch", epoch)
        epoch_start = time.time()
        model.train()
        ####
        # train everything at once
        ####
        regression_train_losses = []
        regression_train_accuracies = []
        regression_train_accuracies2 = []
        regression_start = time.time()
        for batch_idx, sequence_batch in enumerate(train_loader):
            sequence_batch = Variable(sequence_batch, requires_grad=False)
            if use_cuda:
                sequence_batch = sequence_batch.cuda()

            sequence = sequence_batch.squeeze(dim=0)
            subsequences = torch.split(sequence, split_size=15)
            for seq in subsequences:
                batch_size = 1
                seq_len = seq.size()[0]
                seq = seq.view(seq_len, -1).contiguous()
                seq = seq.unsqueeze(dim=1)
                targets = seq[1:]

                optimizer.zero_grad()
                predictions, encode_cost, target_recons = model(seq)
                loss = encode_cost + sum(
                    [sum([F.binary_cross_entropy(input=recon, target=targets[step]) for recon in recons]) for step, recons in
                     enumerate(target_recons[:-1])])
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), .25)
                optimizer.step()
                regression_train_losses.append(np.mean(loss.data.cpu().numpy()))
                accuracies = [F.mse_loss(input=recons[-1], target=targets[step]) for step, recons in enumerate(predictions[:-1])]
                regression_train_accuracies.append(np.mean([acc.data.cpu().numpy() for acc in accuracies]))

                acc = []
                p = [recons[-1] for recons in predictions[:-1]]
                p = torch.cat(p).view(batch_size, seq_len - 1, rest)
                t = targets.view(batch_size, seq_len - 1, rest)
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
            sequence_batch = Variable(sequence_batch, requires_grad=False, volatile=True)
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

        preds, _, _ = model(flat_example)
        preds = torch.stack([flat_example[0]] + preds)
        preds = preds.view(sequence_len + 1, 1, 15, 15)
        save_image(preds.data.cpu(), '_bouncing_sen_{!s}.png'.format(epoch), nrow=10)
        images = [ToPILImage()(pred) for pred in preds.data.cpu()]
        with open('_bouncing_sen_{!s}.gif'.format(epoch), 'wb') as fp:
            images[0].save(fp, save_all=True, append_images=images[1:])

        with open('_bouncing_sen_reg_train.csv', 'a') as f:
            lines = ['{!s},{!s},{!s}\n'.format(loss, acc, acc2) for
                     loss, acc, acc2 in zip(regression_train_losses, regression_train_accuracies, regression_train_accuracies2)]
            for line in lines:
                f.write(line)
        with open('_bouncing_sen_.csv', 'a') as f:
            f.write('{!s},{!s},{!s},{!s},{!s}\n'.format(np.mean(regression_train_losses), np.mean(regression_train_accuracies), np.mean(test_accuracies), np.mean(regression_train_accuracies2), np.mean(test_accuracies2)))

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        print("Epoch took {!s}, estimate {!s} remaining".format(
            make_time_units_string(epoch_time),
            make_time_units_string(np.average(times) * (epochs - 1 - epoch))
        ))
