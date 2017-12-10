"""
CMU motion capture dataset from:
https://github.com/sidsig/NIPS-2014/tree/master/rtrbm/data
"""
from scipy.io import loadmat
import numpy as np
import torch.utils.data as data


class Mocap(data.Dataset):
	def __init__(self, mode='train'):
		super().__init__()
		self.mode = mode

		A = loadmat('MOCAP')
		self.data = A['batchdata']  # size - (3826, 49)
		self.seq_lengths = A['seqlengths'][0]  # [438  260 3128]
		self.seq_starts = np.concatenate(([0], np.cumsum(self.seq_lengths))).astype('int32')[:-1]  # [0  438  698]
		#  split into train/test set per sequence
		pass

	def __getitem__(self, index):
		seq_len = self.seq_lengths[index]
		# train/test split is 80% of the sequence
		train_len = int(seq_len*.8)
		if self.mode == 'train':
			seq_len = train_len
		else:
			seq_len = seq_len - train_len
		seq_start = self.seq_starts[index]
		if self.mode != 'train':
			seq_start = seq_start + train_len
		seq_end = seq_start + seq_len
		return self.data[seq_start:seq_end]

	def __len__(self):
		# 3 sequences in dataset
		return 3
