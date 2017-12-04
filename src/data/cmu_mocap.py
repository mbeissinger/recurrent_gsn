"""
CMU motion capture dataset from:
https://github.com/sidsig/NIPS-2014/tree/master/rtrbm/data
"""
from scipy.io import loadmat
import numpy as np

A = loadmat('MOCAP')
data = A['batchdata']  # size - (3826, 49)
seq_lengths = A['seqlengths'][0]  # [438  260 3128]
seq_starts = np.concatenate(([0], np.cumsum(seq_lengths))).astype('int32')[:-1]  # [0  438  698]
