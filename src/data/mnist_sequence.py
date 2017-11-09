from PIL import Image
from torchvision.datasets import MNIST


class SequencedMNIST(MNIST):
    def __init__(self, root="./datasets", train=True, transform=None, target_transform=None, download=True, sequence=1):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        if self.train:
            self.train_data, self.train_labels = sequence_mnist(self.train_data, self.train_labels, sequence)
        else:
            self.test_data, self.test_labels = sequence_mnist(self.test_data, self.test_labels, sequence)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (images, targets) where targets is list of index of the target class for the sequence.
        """
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


_classes = 10


def sequence_mnist(images, labels, dataset=1):
    def set_xy_indices(x, y, indices):
        x = x.numpy()[indices]
        y = y.numpy()[indices]
        return x, y

    # Find the order of MNIST data for the given sequence id
    ordered_indices = None
    if dataset == 1:
        ordered_indices = dataset1_indices(labels)
    elif dataset == 2:
        ordered_indices = dataset2_indices(labels)
    elif dataset == 3:
        ordered_indices = dataset3_indices(labels)
    elif dataset == 4:
        ordered_indices = dataset4_indices(labels)

    # Put the data sets in order
    if ordered_indices is not None:
        images, labels = set_xy_indices(images, labels, ordered_indices)

    return images, labels


def create_label_pool(labels):
    pool = []
    for _ in range(_classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    return pool


def dataset1_indices(labels):
    # Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset) that makes the numbers go in order 0-9....
    sequence = []
    pool = create_label_pool(labels)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from 0-9 sequencing - missing some class of labels")
    while not stop:
        for i in range(_classes):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence


# order sequentially up then down 0-9-9-0....
def dataset2_indices(labels):
    sequence = []
    pool = create_label_pool(labels)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset2 sequencing - missing some class of labels")
    while not stop:
        for i in list(range(_classes)) + list(range(_classes - 1, -1, -1)):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence


def dataset3_indices(labels):
    sequence = []
    pool = create_label_pool(labels)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset3 sequencing - missing some class of labels")
    a = False
    while not stop:
        for i in range(_classes):
            if not stop:
                n = i
                if i == 1 and a:
                    n = 4
                elif i == 4 and a:
                    n = 8
                elif i == 8 and a:
                    n = 1
                if len(pool[n]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
        a = not a

    return sequence


# extra bits of parity
def dataset4_indices(labels):
    def even(n):
        return n % 2 == 0

    def odd(n):
        return not even(n)

    sequence = []
    pool = create_label_pool(labels)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset4 sequencing - missing some class of labels")
    s = [0, 1, 2]
    sequence.append(pool[0].pop())
    sequence.append(pool[1].pop())
    sequence.append(pool[2].pop())
    while not stop:
        if odd(s[-3]):
            first_bit = (s[-2] - s[-3]) % _classes
        else:
            first_bit = (s[-2] + s[-3]) % _classes
        if odd(first_bit):
            second_bit = (s[-1] - first_bit) % _classes
        else:
            second_bit = (s[-1] + first_bit) % _classes
        if odd(second_bit):
            next_num = (s[-1] - second_bit) % _classes
        else:
            next_num = (s[-1] + second_bit + 1) % _classes

        if len(pool[next_num]) == 0:  # stop the procedure if you are trying to pop from an empty list
            stop = True
        else:
            s.append(next_num)
            sequence.append(pool[next_num].pop())

    return sequence


if __name__ == '__main__':
    print("dataset 1:")
    mnist = SequencedMNIST(sequence=1)
    print(mnist[:50][1])
    print("dataset 2:")
    mnist = SequencedMNIST(sequence=2)
    print(mnist[:50][1])
    print("dataset 3:")
    mnist = SequencedMNIST(sequence=3)
    print(mnist[:50][1])
    print("dataset 4:")
    mnist = SequencedMNIST(sequence=4)
    print(mnist[:50][1])
