from keras.datasets import mnist

_classes = 10


def sequence_mnist(dataset=1):
	def set_xy_indices(x, y, indices):
		x = x[indices]
		y = y[indices]
		return x, y

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Find the order of MNIST data for the given sequence id
	train_ordered_indices, test_ordered_indices = None, None
	if dataset == 1:
		train_ordered_indices = dataset1_indices(y_train)
		test_ordered_indices = dataset1_indices(y_test)
	elif dataset == 2:
		train_ordered_indices = dataset2_indices(y_train)
		test_ordered_indices = dataset2_indices(y_test)
	elif dataset == 3:
		train_ordered_indices = dataset3_indices(y_train)
		test_ordered_indices = dataset3_indices(y_test)
	elif dataset == 4:
		train_ordered_indices = dataset4_indices(y_train)
		test_ordered_indices = dataset4_indices(y_test)

	# Put the data sets in order
	if train_ordered_indices is not None and test_ordered_indices is not None:
		x_train, y_train = set_xy_indices(x_train, y_train, train_ordered_indices)
		x_test, y_test = set_xy_indices(x_test, y_test, test_ordered_indices)

	return (x_train, y_train), (x_test, y_test)


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
	(x_train, y_train), (x_test, y_test) = sequence_mnist(1)
	print(y_train[:50])
	print(y_test[:50])
	print("dataset 2:")
	(x_train, y_train), (x_test, y_test) = sequence_mnist(2)
	print(y_train[:50])
	print(y_test[:50])
	print("dataset 3:")
	(x_train, y_train), (x_test, y_test) = sequence_mnist(3)
	print(y_train[:50])
	print(y_test[:50])
	print("dataset 4:")
	(x_train, y_train), (x_test, y_test) = sequence_mnist(4)
	print(y_train[:50])
	print(y_test[:50])
