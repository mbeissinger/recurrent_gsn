import abc
import warnings
from abc import ABCMeta
from itertools import izip
from tempfile import mkdtemp

import RAB_tools
import cPickle
import glob
import gzip
import h5py
import numpy
import os
import pylab
import pylearn2.datasets.mnist as mnist
import pylearn2.datasets.tfd as tfd
import scipy
import scipy.io
import scipy.io
import scipy.sparse
import socket
import sys
import theano
import theano.sparse
import theano.tensor as T
# import cifar10_wrapper
from PIL import Image
from conditional_nade.tools import image_tiler
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import train_test_split

rng_np = numpy.random.RandomState(1234)


class DataEngine(object):
    def __init__(self, signature, minibatch_size, input_dtype,
                 target_dtype, verbose=True):
        self.verbose = verbose
        self.signature = signature
        self.mode = 0
        self.minibatch_size = minibatch_size
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.n_folds = None
        self._load_dataset()

    def get_dataset(self, which=None, force=False):
        # if force=True, we force to get dataset, not return None.
        if force:
            x, y = self.dp.get_dataset(which=which)
        else:
            if isinstance(self.dp, DataProvider_FitMemoryGroup) or \
                    isinstance(self.dp, DataProvider_leaveKout):
                x = None
                y = None
            else:

                x, y = self.dp.get_dataset(which=which)
        return x, y

    def get_dataset_range(self):
        # return a list of [(dim1_min, dim1_max), (dim2_min, dim2_max), ...]
        xs = []
        names = ['train', 'valid', 'test']
        for i in range(3):
            x, _ = self.get_dataset(which=names[i], force=True)
            xs.append(x)
        xs = numpy.concatenate(xs, axis=0)

        mins = xs.min(axis=0)
        maxs = xs.max(axis=0)
        rval = zip(mins, maxs)

        return rval

    def get_theano_shared(self, which=None):
        inputs, labels = self.dp.get_theano_shared(which=which)
        return inputs, labels

    def get_a_minibatch_idx(self, which=None):
        start_idx, end_idx, x, y = self.dp.get_a_minibatch_idx(which=which)
        return start_idx, end_idx, x, y

    def get_a_minibatch(self, which=None):
        # this is called when using DataProvider_OnTheFly
        self.db.get_a_minibatch(which=which)

    def prepare_fold(self, fold_idx):
        if fold_idx == 0:
            pass
        else:
            self.dp.next_fold()

    def _load_dataset(self):
        if self.signature == 'MNIST':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(shuffle=True)
        if self.signature == 'manifold':
            (train, train_label, valid,
             valid_label, test, test_label) = load_manifold()
        elif self.signature == 'MNIST_binary':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=True)
        elif self.signature == '3Dcorkscrew':
            (train, train_label, valid,
             valid_label, test, test_label) = load_3D_corkscrew()
        elif self.signature == '2Dspiral':
            (train, train_label, valid,
             valid_label, test, test_label) = load_2D_curves(curve='spiral')

        elif self.signature == '2Dstraightline':
            (train, train_label, valid,
             valid_label, test, test_label) = load_2D_curves(curve='straightline')

        elif self.signature == '2Dcircle':
            (train, train_label, valid,
             valid_label, test, test_label) = load_2D_curves(curve='circle')


        elif self.signature == 'MNIST_binary_xy':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=True, xy=True)
        elif self.signature == 'MNIST_binary_5k_1k_1k':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=True,
                                                         standard_split=False)
        elif 'MNIST_binary_5k_1k_1k_permute' in self.signature:
            seed = int(self.signature.split('_')[-1])
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=True,
                                                         permute_bits_seed=seed,
                                                         standard_split=False)
        elif self.signature == 'MNIST_binary_6k_1k':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=True,
                                                         standard_split=True)
        elif self.signature == 'MNIST_continuous_6k_1k':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=False,
                                                         standard_split=True)
        elif self.signature == 'MNIST_continuous':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist(binary=False)

        elif self.signature == 'MNIST_binary_14_14':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mnist_scaled(binarize=True)

        elif self.signature == 'debug':
            (train, train_label, valid,
             valid_label, test, test_label) = load_debug_dataset()

        elif self.signature == 'TFD_unsupervised':
            (train, train_label, valid,
             valid_label, test, test_label) = load_TFD(style='unsupervised')

        elif self.signature == 'TFD_supervised':
            (train, train_label, valid,
             valid_label, test, test_label) = load_TFD(style='supervised')

        elif self.signature == 'TFD_unsupervised_and_supervised':
            (train, train_label, valid,
             valid_label, test, test_label) = load_TFD(style='unsupervised')

        elif self.signature == 'GRO_Fun_744':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='normal',
                                                                   which='744')
        elif self.signature == 'GRO_Fun_1008':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='normal',
                                                                   which='1008')

        elif self.signature == 'GRO_Fun_744_reverse':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='reverse',
                                                                   which='744')
        elif self.signature == 'GRO_Fun_744_reverse_diff':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='reverse',
                                                                   which='744', use_differential_targets=True)

        elif self.signature == 'GRO_Fun_744_diff':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='normal',
                                                                   which='744', use_differential_targets=True)

        elif self.signature == 'GRO_Fun_1008_reverse_diff':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='reverse',
                                                                   which='1008', use_differential_targets=True)

        elif self.signature == 'GRO_Fun_1008_reverse':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_fun_labeled(mode='reverse',
                                                                   which='1008')
        elif self.signature == 'facetube_pairs_TS':
            (get_train_fn, get_valid_fn,
             get_test_fn, filter_method, chunk_size) = load_facetube_TS()
            self.mode = 2

        elif 'brodatz_patches' in self.signature:
            which_one = RAB_tools.find_all_numbers_in_string(self.signature)[-1]
            (train, train_label, valid,
             valid_label, test, test_label) = load_brodatz_patches(which_one)

        elif 'brodatz_conv' in self.signature:
            which_pic = RAB_tools.find_all_numbers_in_string(self.signature)[-1]
            image = load_brodatz_whole(which_pic)
            self.mode = 9

        elif self.signature == 'MNIST_Batches':
            (file_prefix, batch_idx_train,
             batch_idx_valid, batch_idx_test) = load_MNIST_batches()
            self.mode = 1
            self.c_ordered = False

        elif self.signature == 'GRO_Fun_Batches':
            (file_prefix, batch_idx_train,
             batch_idx_valid, batch_idx_test) = load_GRO_fun_batches()
            self.mode = 1
            self.c_ordered = True

        elif self.signature == 'GRO_Fun_Batches_1008':
            (file_prefix, batch_idx_train,
             batch_idx_valid, batch_idx_test) = load_GRO_fun_batches_1008()
            self.mode = 1
            self.c_ordered = True

        elif self.signature == 'GRO_Win':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_winner()

        elif self.signature == 'GRO_Kill_620':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_kill_ratio(which='620')

        elif self.signature == 'GRO_Kill_Calibration':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_kill_ratio_calibration()

        elif self.signature == 'GRO_Kill_840':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_kill_ratio(which='840')

        elif self.signature == 'GRO_Kill_620_reverse':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_kill_ratio(mode='reverse',
                                                                  which='620')

        elif self.signature == 'GRO_Kill_840_reverse':
            (train, train_label, valid,
             valid_label, test, test_label) = load_GRO_kill_ratio(mode='reverse',
                                                                  which='840')
        elif self.signature == 'mighty_diff':
            (train, train_label, valid,
             valid_label, test, test_label) = load_mighty_quest()

        elif self.signature == 'CIFAR10':
            (train, train_label, valid,
             valid_label, test, test_label) = load_cifar10_raw()

        elif self.signature == 'face_tubes':
            (train, train_label, valid,
             valid_label, test, test_label) = load_face_tubes()

        elif self.signature == 'face_tubes_batches':
            (file_prefix, batch_idx_train,
             batch_idx_valid, batch_idx_test) = load_face_tubes_batches()
            self.mode = 1
            self.c_ordered = True

        elif self.signature == 'shifted_images':
            (train, train_label, valid,
             valid_label, test, test_label) = load_shifted_images()

        elif self.signature == 'biodesix_p0':
            (train, train_label, test, test_label) = load_biodesix(task=[0])
            self.mode = 6

        elif self.signature == 'biodesix_p1':
            (train, train_label, test, test_label) = load_biodesix(task=[1])
            self.mode = 6

        elif self.signature == 'biodesix_p2':
            (train, train_label, test, test_label) = load_biodesix(task=[2])
            self.mode = 6

        elif self.signature == 'biodesix_p3':
            (train, train_label, test, test_label) = load_biodesix(task=[3])
            self.mode = 6

        elif self.signature == 'biodesix_p4':
            (train, train_label, test, test_label) = load_biodesix(task=[4])
            self.mode = 6

        elif self.signature == 'biodesix_p5':
            (train, train_label, test, test_label) = load_biodesix(task=[5])
            self.mode = 6

        elif self.signature == 'biodesix_p1_mix':
            (train, train_label, test, test_label) = load_biodesix(task=[1], mix=True)
            self.mode = 6


        elif self.signature == 'biodesix_p2_mix':
            (train, train_label, test, test_label) = load_biodesix(task=[2], mix=True)
            self.mode = 6

        elif self.signature == 'biodesix_p3_mix':
            (train, train_label, test, test_label) = load_biodesix(task=[3], mix=True)
            self.mode = 6

        elif self.signature == 'biodesix_p4_mix':
            (train, train_label, test, test_label) = load_biodesix(task=[4], mix=True)
            self.mode = 6

        elif self.signature == 'biodesix_p5_mix':
            (train, train_label, test, test_label) = load_biodesix(task=[5], mix=True)
            self.mode = 6


        elif self.signature == 'test_mix':
            (train, train_label, test, test_label) = load_test_mix()
            self.mode = 6

        elif self.signature == 'biodesix_p1_p0':
            (train, train_label, test, test_label) = load_biodesix(task=[0, 1])
            self.mode = 4

        elif self.signature == 'biodesix_all_6':
            (train, train_label, test, test_label) = load_biodesix(task=[0, 1, 2, 3, 4, 5])
            self.mode = 7

        elif self.signature == 'biodesix_all_supervised':
            (train, train_label, test, test_label) = load_biodesix(task=[1, 2, 3, 4, 5])
            self.mode = 6

        elif self.signature == 'biodesix_all_pretrain':
            (train, train_label, test, test_label) = load_biodesix(task=[0, 1, 2, 3, 4, 5],
                                                                   pretrain=True)
            self.mode = 6
        elif self.signature == 'mnist_0_3_4_5_8':
            (train, train_label, test, test_label) = load_mnist_multitasking()
            self.mode = 7

        elif self.signature == 'stackexchange_small':
            (train, train_label,
             valid, valid_label,
             test, test_label) = load_stackexchange(small=True)
            self.n_folds = 2
            self.mode = 0

        else:
            NotImplementedError('unsupported dataset signature!')

        if self.mode == 1:
            self._create_dp_scalable(file_prefix, batch_idx_train,
                                     batch_idx_valid, batch_idx_test)
        elif self.mode == 0:
            self._create_dp_multiple(train, train_label, valid,
                                     valid_label, test, test_label)

        elif self.mode == 3:
            self._create_dp_leaveKout(train, train_label, test, test_label, special=False)

        elif self.mode == 4:
            self._create_dp_leaveKout(train, train_label, test, test_label, special=True)

        elif self.mode == 5:
            self._create_dp_KFold(train, train_label, test, test_label, special=True, k=10)

        elif self.mode == 6:
            self._create_dp_KFold(train, train_label, test, test_label,
                                  special=False, k=10)
        elif self.mode == 7:
            self._create_dp_KFold(train, train_label, test, test_label,
                                  special=True, k=10, resample_trainset=True)

        elif self.mode == 8:
            self._create_dp_KFold(train, train_label, test, test_label,
                                  special=True, k=5, resample_trainset=True)

        elif self.mode == 9:
            self._create_dp_onthefly(image)

        elif self.mode == 2:
            self._create_dp_online(get_train_fn, get_valid_fn,
                                   get_test_fn, filter_method, chunk_size)

        else:
            raise NotImplementedError('mode is not supported!')

    def _create_dp_onthefly(self, image):
        # fn is able to generate patches
        self.dp = DataProvider_OnTheFly(
            source=image,
            minibatch_size=self.minibatch_size
        )

    def _create_dp_online(self, get_train_fn, get_valid_fn,
                          get_test_fn, filter_method, chunk_size):
        """
        fns should return x, y 
        """
        data_source = TimeSeriesData(get_train_fn, get_valid_fn,
                                     get_test_fn, filter_method, chunk_size)
        self.dp = DataProvider_Online(signature=self.signature,
                                      data_source=data_source,
                                      input_dtype=self.input_dtype,
                                      target_dtype=self.target_dtype,
                                      verbose=self.verbose,
                                      minibatch_size=self.minibatch_size)

    def _create_dp_scalable(self, file_prefix, batch_idx_train,
                            batch_idx_valid, batch_idx_test):

        self.dp = DataProvider_Scalable(
            signature=self.signature,
            prefix=file_prefix,
            train_batch_idx=batch_idx_train,
            valid_batch_idx=batch_idx_valid,
            test_batch_idx=batch_idx_test,
            minibatch_size=self.minibatch_size,
            c_ordered=self.c_ordered,
            input_dtype=self.input_dtype,
            verbose=self.verbose,
            target_dtype=self.target_dtype
        )

    def _create_dp_leaveKout(self, train, train_label, test, test_label, special):

        self.dp = DataProvider_leaveKout(train, train_label, test, test_label,
                                         self.minibatch_size, self.verbose, special)
        self.n_folds = self.dp.n_folds

    def _create_dp_KFold(self, train, train_label, test, test_label,
                         special, k, resample_trainset=False):

        self.dp = DataProvider_KFold(train, train_label, test, test_label,
                                     self.minibatch_size, self.verbose,
                                     special, k, resample_trainset=resample_trainset)
        self.n_folds = self.dp.n_folds

    def _create_dp_multiple(self, train, train_label, valid,
                            valid_label, test, test_label):
        train_input_provider = DataProvider_FitMemory(
            dataset=train,
            dataset_name='trainset inputs',
            dtype=self.input_dtype,
            minibatch_size=self.minibatch_size, verbose=self.verbose)

        train_label_provider = DataProvider_FitMemory(
            dataset=train_label,
            dtype=self.target_dtype,
            dataset_name='train labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        valid_input_provider = DataProvider_FitMemory(
            dataset=valid,
            dtype=self.input_dtype,
            dataset_name='valid inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        valid_label_provider = DataProvider_FitMemory(
            dataset=valid_label,
            dtype=self.target_dtype,
            dataset_name='valid labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        test_input_provider = DataProvider_FitMemory(
            dataset=test,
            dtype=self.input_dtype,
            dataset_name='test inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        test_label_provider = DataProvider_FitMemory(
            dataset=test_label,
            dtype=self.target_dtype,
            dataset_name='test labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.dp = DataProvider_FitMemoryGroup(train_input_provider,
                                              train_label_provider,
                                              valid_input_provider,
                                              valid_label_provider,
                                              test_input_provider,
                                              test_label_provider,
                                              )


class DataProvider:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def _refresh(self, which=None):
        return

    @abc.abstractmethod
    def get_a_minibatch_idx(self, which=None):
        return

    @abc.abstractmethod
    def get_dataset(self, which=None):
        return

    @abc.abstractmethod
    def next_fold(self, which=None):
        return

    @abc.abstractmethod
    def get_minibatch_size(self):
        return

    @abc.abstractmethod
    def get_theano_shared(self, which=None):
        # only calable when constructing theano_fns
        return

    def get_a_minibatch(self, which=None):
        raise NotImplementedError()


class DataProvider_OnTheFly(DataProvider):
    def __init__(self, source, minibatch_size):
        # source is an image of class Brodatz
        self.minibatch_size = minibatch_size
        self.source = source
        self.minibatch_counter = minibatch_size

    def get_a_minibatch(self, which, patch_size):
        assert self.source is not None

        minibatch = self.source.get_minibatch_train(
            minibatch_size=self.minibatch_size,
            patch_size=patch_size
        )
        self.minibatch_counter += 1
        return minibatch

    def get_test_image(self):
        return self.source.get_test_image()

    def _refresh(self, which=None):
        raise NotImplementedError()

    def get_a_minibatch_idx(self, which=None):
        raise NotImplementedError()

    def get_dataset(self, which=None):
        raise NotImplementedError()
        return

    def next_fold(self, which=None):
        raise NotImplementedError()
        return

    def get_minibatch_size(self):
        self.minibatch_size

    def get_theano_shared(self, which=None):
        return None, None


class DataProvider_KFold(DataProvider):
    def __init__(self, train, train_label, test, test_label,
                 minibatch_size, verbose, special, k,
                 retrain=True, resample_trainset=False):
        '''
        KFold will divide train into K folds. The extra K+1 fold is the
        retrain fold that uses all train as train, test as valid.
        '''

        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.special = special
        self.k = k
        self.original_train = train
        self.original_test = test
        self.resample_trainset = resample_trainset

        if special:

            assert train_label.shape[1] == 2
            assert test_label.shape[1] == 2

            print
            'special K Fold with missing labels'
            assert numpy.sum(train_label[:, 0] == -1) != 0
            train_idx_unsup = train_label[:, 0] == -1
            self.train_unsup = train[train_idx_unsup]
            self.train_unsup_label = train_label[train_idx_unsup]
            train_idx_sup = train_label[:, 0] != -1
            self.train_sup = train[train_idx_sup]
            self.train_sup_label = train_label[train_idx_sup]
        else:

            print
            'ordinary leave k out without misssing labels'
            self.train_unsup = None
            self.train_unsup_label = None
            self.train_sup = train
            self.train_sup_label = train_label

        self.train = self.train_sup
        self.train_label = self.train_sup_label
        self.test = test
        self.test_label = test_label
        self.input_dtype = 'float32'
        self.target_dtype = 'int32'

        loo = KFold(self.train_label.shape[0], n_folds=self.k, random_state=1234)
        self.loo_indices = []
        for i, j in loo:
            indices = [i, j]
            self.loo_indices.append(indices)
        # the last fold is added because we need retraining on all trainset
        self.n_folds = len(self.loo_indices) + 1
        self.current_fold = 0

        if special:
            train_x = numpy.concatenate([self.train[self.loo_indices[0][0]],
                                         self.train_unsup], axis=0)
            train_y = numpy.concatenate([self.train_label[self.loo_indices[0][0]],
                                         self.train_unsup_label], axis=0)
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]
        else:
            train_x = self.train[self.loo_indices[0][0]]
            train_y = self.train_label[self.loo_indices[0][0]]
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]

        self.build_data_providers(train_x=train_x,
                                  train_y=train_y,
                                  valid_x=self.train[self.loo_indices[0][1]],
                                  valid_y=self.train_label[self.loo_indices[0][1]],
                                  test_x=test,
                                  test_y=test_label,
                                  )

    def build_data_providers(self, train_x, train_y, valid_x,
                             valid_y, test_x, test_y):
        # if resample, then resample/rebalance train_x, and adjust train_y accordingly
        if 0 and self.resample_trainset:
            print
            'resampling trainset...'
            new_train_x, new_train_y = RAB_tools.resample(train_x, train_y)
            train_x = new_train_x.astype(train_x.dtype)
            train_y = new_train_y.astype(train_y.dtype)

        self.train_input_provider = DataProvider_FitMemory(
            dataset=train_x,
            dataset_name='trainset inputs',
            dtype=self.input_dtype,
            minibatch_size=self.minibatch_size, verbose=self.verbose)

        self.train_label_provider = DataProvider_FitMemory(
            dataset=train_y,
            dtype=self.target_dtype,
            dataset_name='train labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.valid_input_provider = DataProvider_FitMemory(
            dataset=valid_x,
            dtype=self.input_dtype,
            dataset_name='valid inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.valid_label_provider = DataProvider_FitMemory(
            dataset=valid_y,
            dtype=self.target_dtype,
            dataset_name='valid labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.test_input_provider = DataProvider_FitMemory(
            dataset=test_x,
            dtype=self.input_dtype,
            dataset_name='test inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.test_label_provider = DataProvider_FitMemory(
            dataset=test_y,
            dtype=self.target_dtype,
            dataset_name='test labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

    def next_fold(self):
        self.current_fold += 1
        assert self.current_fold < self.n_folds
        if self.current_fold == self.n_folds - 1:
            print
            'last fold'
            # this is the last fold,
            # use full train as train, test as valid, test as test
            if self.special:
                train_x = numpy.concatenate(
                    [self.train,
                     self.train_unsup], axis=0)
                train_y = numpy.concatenate(
                    [self.train_label,
                     self.train_unsup_label], axis=0)
                train_x, idx = RAB_tools.shuffle_dataset(train_x)
                train_y = train_y[idx]
            else:
                train_x = self.train
                train_y = self.train_label

            self.build_data_providers(
                train_x=train_x,
                train_y=train_y,
                valid_x=self.test,
                valid_y=self.test_label,
                test_x=self.test,
                test_y=self.test_label)

        else:
            # not the full retrain fold
            if self.special:

                train_x = numpy.concatenate(
                    [self.train[self.loo_indices[self.current_fold][0]],
                     self.train_unsup], axis=0)
                train_y = numpy.concatenate(
                    [self.train_label[self.loo_indices[self.current_fold][0]],
                     self.train_unsup_label], axis=0)
                train_x, idx = RAB_tools.shuffle_dataset(train_x)
                train_y = train_y[idx]
            else:
                train_x = self.train[self.loo_indices[self.current_fold][0]]
                train_y = self.train_label[self.loo_indices[self.current_fold][0]]
                train_x, idx = RAB_tools.shuffle_dataset(train_x)
                train_y = train_y[idx]

            self.build_data_providers(
                train_x=train_x,
                train_y=train_y,
                valid_x=self.train[self.loo_indices[self.current_fold][1]],
                valid_y=self.train_label[self.loo_indices[self.current_fold][1]],
                test_x=self.test,
                test_y=self.test_label)

    def get_a_minibatch_idx(self, which=None):
        if which == 'train':
            start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
        elif which == 'valid':
            start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
        elif which == 'test':
            start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()

        return start_idx, end_idx, None, None

    def _refresh(self, which=None):
        pass

    def get_dataset(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_dataset()
            y = self.train_label_provider.get_dataset()
        if which == 'valid':
            x = self.valid_input_provider.get_dataset()
            y = self.valid_label_provider.get_dataset()
        if which == 'test':
            x = self.test_input_provider.get_dataset()
            y = self.test_label_provider.get_dataset()
        return x, y

    def get_theano_shared(self, which=None):

        if which == 'train':
            x = self.train_input_provider.get_theano_shared()
            y = self.train_label_provider.get_theano_shared()
        if which == 'valid':
            x = self.valid_input_provider.get_theano_shared()
            y = self.valid_label_provider.get_theano_shared()
        if which == 'test':
            x = self.test_input_provider.get_theano_shared()
            y = self.test_label_provider.get_theano_shared()

        return x, y

    def get_minibatch_size(self):
        NotImplementedError('You should not reach here')


class DataProvider_leaveKout(DataProvider):
    def __init__(self, train, train_label, test, test_label,
                 minibatch_size, verbose, special):
        # if special, then leave k out is done on the labeled part,
        # unlabeled part is simply added into trainset of each each fold
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.special = special
        assert train_label.shape[1] == 2
        assert test_label.shape[1] == 2

        if special:
            print
            'special leave k out with missing labels'
            assert numpy.sum(train_label[:, 0] == -1) != 0
            train_idx_unsup = train_label[:, 0] == -1
            self.train_unsup = train[train_idx_unsup]
            self.train_unsup_label = train_label[train_idx_unsup]
            train_idx_sup = train_label[:, 0] != -1
            self.train_sup = train[train_idx_sup]
            self.train_sup_label = train_label[train_idx_sup]
        else:
            print
            'ordinary leave k out without misssing labels'
            self.train_unsup = []
            self.train_unsup_label = []
            self.train_sup = train
            self.train_sup_label = train_label

        self.train = self.train_sup
        self.train_label = self.train_sup_label
        self.test = test
        self.test_label = test_label
        self.input_dtype = 'float32'
        self.target_dtype = 'int32'

        loo = LeaveOneOut(self.train_label.shape[0])
        self.loo_indices = []
        for i, j in loo:
            indices = [i, j]
            self.loo_indices.append(indices)

        self.n_folds = len(self.loo_indices)
        self.current_fold = 0

        if special:
            train_x = numpy.concatenate([self.train[self.loo_indices[0][0]],
                                         self.train_unsup], axis=0)
            train_y = numpy.concatenate([self.train_label[self.loo_indices[0][0]],
                                         self.train_unsup_label], axis=0)
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]
        else:
            train_x = self.train[self.loo_indices[0][0]]
            train_y = self.train_label[self.loo_indices[0][0]]
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]

        self.build_data_providers(train_x=train_x,
                                  train_y=train_y,
                                  valid_x=self.train[self.loo_indices[0][1]],
                                  valid_y=self.train_label[self.loo_indices[0][1]],
                                  test_x=test,
                                  test_y=test_label)

    def build_data_providers(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        self.train_input_provider = DataProvider_FitMemory(
            dataset=train_x,
            dataset_name='trainset inputs',
            dtype=self.input_dtype,
            minibatch_size=self.minibatch_size, verbose=self.verbose)

        self.train_label_provider = DataProvider_FitMemory(
            dataset=train_y,
            dtype=self.target_dtype,
            dataset_name='train labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.valid_input_provider = DataProvider_FitMemory(
            dataset=valid_x,
            dtype=self.input_dtype,
            dataset_name='valid inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.valid_label_provider = DataProvider_FitMemory(
            dataset=valid_y,
            dtype=self.target_dtype,
            dataset_name='valid labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.test_input_provider = DataProvider_FitMemory(
            dataset=test_x,
            dtype=self.input_dtype,
            dataset_name='test inputs', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

        self.test_label_provider = DataProvider_FitMemory(
            dataset=test_y,
            dtype=self.target_dtype,
            dataset_name='test labels', minibatch_size=self.minibatch_size,
            verbose=self.verbose)

    def next_fold(self):
        assert self.current_fold < self.n_folds
        self.current_fold += 1
        if self.special:
            train_x = numpy.concatenate(
                [self.train[self.loo_indices[self.current_fold][0]],
                 self.train_unsup], axis=0)
            train_y = numpy.concatenate(
                [self.train_label[self.loo_indices[self.current_fold][0]],
                 self.train_unsup_label], axis=0)
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]
        else:
            train_x = self.train[self.loo_indices[0][0]]
            train_y = self.train_label[self.loo_indices[0][0]]
            train_x, idx = RAB_tools.shuffle_dataset(train_x)
            train_y = train_y[idx]

        self.build_data_providers(
            train_x=train_x,
            train_y=train_y,
            valid_x=self.train[self.loo_indices[self.current_fold][1]],
            valid_y=self.train_label[self.loo_indices[self.current_fold][1]],
            test_x=self.test,
            test_y=self.test_label)

    def get_a_minibatch_idx(self, which=None):
        if which == 'train':
            start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
        elif which == 'valid':
            start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
        elif which == 'test':
            start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()

        return start_idx, end_idx, None, None

    def _refresh(self, which=None):
        pass

    def get_dataset(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_dataset()
            y = self.train_label_provider.get_dataset()
        if which == 'valid':
            x = self.valid_input_provider.get_dataset()
            y = self.valid_label_provider.get_dataset()
        if which == 'test':
            x = self.test_input_provider.get_dataset()
            y = self.test_label_provider.get_dataset()
        return x, y

    def get_theano_shared(self, which=None):

        if which == 'train':
            x = self.train_input_provider.get_theano_shared()
            y = self.train_label_provider.get_theano_shared()
        if which == 'valid':
            x = self.valid_input_provider.get_theano_shared()
            y = self.valid_label_provider.get_theano_shared()
        if which == 'test':
            x = self.test_input_provider.get_theano_shared()
            y = self.test_label_provider.get_theano_shared()

        return x, y

    def get_minibatch_size(self):
        NotImplementedError('You should not reach here')


class DataProvider_FitMemoryGroup(DataProvider):
    def __init__(self, train_input_provider, train_label_provider,
                 valid_input_provider, valid_label_provider,
                 test_input_provider, test_label_provider):

        self.train_input_provider = train_input_provider
        self.train_label_provider = train_label_provider
        self.valid_input_provider = valid_input_provider
        self.valid_label_provider = valid_label_provider
        self.test_input_provider = test_input_provider
        self.test_label_provider = test_label_provider

    def next_fold(self):
        train_x = self.train_input_provider.get_dataset()
        train_y = self.train_label_provider.get_dataset()
        valid_x = self.valid_input_provider.get_dataset()
        valid_y = self.valid_label_provider.get_dataset()
        test_x = self.test_input_provider.get_dataset()
        test_y = self.test_label_provider.get_dataset()

        verbose = self.train_input_provider.verbose

        minibatch_size = self.train_input_provider.get_minibatch_size()

        if scipy.sparse.issparse(train_x):
            train_x = scipy.sparse.vstack((train_x, valid_x))
        else:
            train_x = numpy.concatenate((train_x, valid_x), axis=0)
        if scipy.sparse.issparse(train_y):
            train_y = scipy.sparse.vstack((train_y, valid_y))
        else:
            train_y = numpy.concatenate((train_y, valid_y), axis=0)
        valid_x = test_x
        valid_y = test_y

        print('\rPreparing dataset for the next fold...')

        self.train_input_provider = DataProvider_FitMemory(
            dataset=train_x, dtype=train_x.dtype,
            dataset_name='trainset inputs', minibatch_size=minibatch_size,
            verbose=verbose)

        self.train_label_provider = DataProvider_FitMemory(
            dataset=train_y, dtype=train_y.dtype,
            dataset_name='train labels', minibatch_size=minibatch_size,
            verbose=verbose)

        self.valid_input_provider = DataProvider_FitMemory(
            dataset=valid_x, dtype=valid_x.dtype,
            dataset_name='valid inputs', minibatch_size=minibatch_size,
            verbose=verbose)

        self.valid_label_provider = DataProvider_FitMemory(
            dataset=valid_y, dtype=valid_y.dtype,
            dataset_name='valid labels', minibatch_size=minibatch_size,
            verbose=verbose)

    def _refresh(self, which=None):
        pass

    def get_a_minibatch_idx(self, which=None):
        if which == 'train':
            start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
        elif which == 'valid':
            start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
        elif which == 'test':
            start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()

        return start_idx, end_idx, None, None

    def get_dataset(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_dataset()
            y = self.train_label_provider.get_dataset()
        if which == 'valid':
            x = self.valid_input_provider.get_dataset()
            y = self.valid_label_provider.get_dataset()
        if which == 'test':
            x = self.test_input_provider.get_dataset()
            y = self.test_label_provider.get_dataset()
        return x, y

    def get_theano_shared(self, which=None):

        if which == 'train':
            x = self.train_input_provider.get_theano_shared()
            y = self.train_label_provider.get_theano_shared()
        if which == 'valid':
            x = self.valid_input_provider.get_theano_shared()
            y = self.valid_label_provider.get_theano_shared()
        if which == 'test':
            x = self.test_input_provider.get_theano_shared()
            y = self.test_label_provider.get_theano_shared()

        return x, y

    def get_minibatch_size(self):
        NotImplementedError('You should not reach here')


class DataProvider_FitMemory(DataProvider):
    # very general worker that takes a dataset and gives out minibatches
    def __init__(self, dataset, dataset_name, dtype, minibatch_size, verbose):
        # assume dataset is a matrix
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.dtype = dtype
        self.verbose = verbose
        try:
            assert self.dtype == self.dataset.dtype
        except AssertionError:
            raise AssertionError('dataset dtype is not consistent')

        self.n_samples = self.dataset.shape[0]
        self.minibatch_size = minibatch_size

        self.current_minibatch = None

        if self.minibatch_size > self.n_samples:
            warnings.warn('minibatch size is bigger than the dataset')

        self.n_minibatches = self.n_samples / self.minibatch_size
        self.leftover_n_samples = self.n_samples % self.minibatch_size
        if self.leftover_n_samples != 0:
            warnings.warn('The last minibatch will have %d instead of %d samples'
                          % (self.leftover_n_samples, self.minibatch_size))

        self.minibatch_counter = 0

    def _refresh(self, which=None):
        # print ''
        # print 'Refreshing...'
        self.minibatch_counter = 0

    def get_a_minibatch(self):
        if self.minibatch_counter < self.n_minibatches:
            start_idx = self.minibatch_counter * self.minibatch_size
            end_idx = (self.minibatch_counter + 1) * self.minibatch_size
            idx = xrange(start_idx, end_idx)

            self.current_minibatch = self.dataset[idx, :]
            self.minibatch_counter += 1

        elif (self.minibatch_counter == self.n_minibatches and
                      self.leftover_n_samples != 0):
            idx = -1 * self.leftover_n_samples
            self.current_minibatch = self.dataset[idx:]
            self.minibatch_counter += 1

        else:
            # end of the current minibatch passing.
            self.current_minibatch = None

        return self.current_minibatch

    def get_a_minibatch_idx(self, which=None):
        # passing minibatches is not efficient, passing indices instead
        if self.minibatch_counter < self.n_minibatches:
            start_idx = self.minibatch_counter * self.minibatch_size
            end_idx = (self.minibatch_counter + 1) * self.minibatch_size
            idx = xrange(start_idx, end_idx)

            # self.current_minibatch = self.dataset[idx, :]
            self.minibatch_counter += 1
            if self.verbose:
                sys.stdout.write('\rProcessing %5d/%5d minibatches' % (
                    self.minibatch_counter, self.n_minibatches))
                sys.stdout.flush()

        elif (self.minibatch_counter == self.n_minibatches and
                      self.leftover_n_samples != 0):

            idx = self.minibatch_counter * self.minibatch_size
            # self.current_minibatch = self.dataset[self.minibatch_counter*self.minibatch_size:]
            self.minibatch_counter += 1
            start_idx = idx
            end_idx = self.n_samples

        else:

            # end of the current minibatch passing.
            idx = None
            self.current_minibatch = None
            start_idx = None
            end_idx = None
            self._refresh()

        return start_idx, end_idx

    def get_dataset(self, which=None):
        return self.dataset

    def get_theano_shared(self, which=None, ):
        if scipy.sparse.issparse(self.dataset):
            rval = theano.sparse.shared(self.dataset)
        else:
            rval = theano.shared(numpy.asarray(self.dataset, dtype=self.dtype))
        return rval

    def next_fold(self):
        pass

    def get_minibatch_size(self):
        return self.minibatch_size


class DataProvider_Online(DataProvider):
    """
    this class deals with mainly time series data where consecutive frames are generated
    in an online fashion
    """

    def __init__(self, signature, data_source, minibatch_size, verbose,
                 input_dtype, target_dtype):
        self.signature = signature
        self.data_source = data_source
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype

        self._preload_first_batch()

    def _preload_first_batch(self):
        print
        'Preloading the trainset and validset'
        self._refresh_trainset()
        self._refresh_validset()
        self._refresh_testset()

    def _refresh(self, which=None):
        if which == 'train':
            self._refresh_trainset()
        elif which == 'valid':
            self._refresh_validset()
        elif which == 'test':
            self._refresh_testset()
        elif which == 'all':
            self._refresh_trainset()
            self._refresh_validset()
            self._refresh_testset()
        else:
            NotImplementedError('which=%s is not supported!' % (which))

    def _refresh_trainset(self):
        print
        'refreshing trainset...'
        train_x, train_y = self.data_source.get_data_batch('train')
        self.train_input_provider = DataProvider_FitMemory(train_x,
                                                           self.signature, self.input_dtype,
                                                           minibatch_size=self.minibatch_size,
                                                           verbose=self.verbose
                                                           )
        self.train_label_provider = DataProvider_FitMemory(train_y,
                                                           self.signature, self.target_dtype,
                                                           verbose=self.verbose,
                                                           minibatch_size=self.minibatch_size)

    def _refresh_validset(self):
        print
        'refreshing validset...'
        valid_x, valid_y = self.data_source.get_data_batch('valid')
        self.valid_input_provider = DataProvider_FitMemory(valid_x,
                                                           self.signature, self.input_dtype,
                                                           minibatch_size=self.minibatch_size,
                                                           verbose=self.verbose
                                                           )
        self.valid_label_provider = DataProvider_FitMemory(valid_y,
                                                           self.signature, self.target_dtype,
                                                           verbose=self.verbose,
                                                           minibatch_size=self.minibatch_size)

    def _refresh_testset(self):
        print
        'refreshing testset...'
        test_x, test_y = self.data_source.get_data_batch('test')
        self.test_input_provider = DataProvider_FitMemory(test_x,
                                                          self.signature, self.input_dtype,
                                                          minibatch_size=self.minibatch_size,
                                                          verbose=self.verbose
                                                          )
        self.test_label_provider = DataProvider_FitMemory(test_y,
                                                          self.signature, self.target_dtype,
                                                          verbose=self.verbose,
                                                          minibatch_size=self.minibatch_size)

    def get_a_minibatch_idx_train(self):
        refresh_theano_shared = False

        start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.train_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            print
            '\nDataProvider_Online: loading the next train chunk...'
            train_x, train_y = self.data_source.get_data_batch('train')
            self.train_input_provider = DataProvider_FitMemory(train_x,
                                                               self.signature,
                                                               self.input_dtype,
                                                               verbose=self.verbose,
                                                               minibatch_size=self.minibatch_size)
            self.train_label_provider = DataProvider_FitMemory(
                train_y,
                self.signature,
                self.target_dtype,
                verbose=self.verbose,
                minibatch_size=self.minibatch_size)

            start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
            start_idx, end_idx = self.train_label_provider.get_a_minibatch_idx()
            refresh_theano_shared = True

        if refresh_theano_shared:
            x = train_x
            y = train_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx_valid(self):
        refresh_theano_shared = False

        start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.valid_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            # print 'loading the next data batch...'
            print
            '\nDataProvider_Online: loading the next valid chunk...'
            valid_x, valid_y = self.data_source.get_data_batch('train')

            self.valid_input_provider = DataProvider_FitMemory(valid_x,
                                                               self.signature,
                                                               self.input_dtype,
                                                               verbose=self.verbose,
                                                               minibatch_size=self.minibatch_size)
            self.valid_label_provider = DataProvider_FitMemory(
                valid_y,
                self.signature,
                self.target_dtype,
                verbose=self.verbose,
                minibatch_size=self.minibatch_size)

            start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
            start_idx, end_idx = self.valid_label_provider.get_a_minibatch_idx()
            refresh_theano_shared = True

        if refresh_theano_shared:
            x = valid_x
            y = valid_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx_test(self):
        refresh_theano_shared = False

        start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.test_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            print
            '\nDataProvider_Online: loading the next test chunk...'
            test_x, test_y = self.data_source.get_data_batch('test')

            self.test_input_provider = DataProvider_FitMemory(test_x,
                                                              self.signature,
                                                              self.input_dtype,
                                                              verbose=self.verbose,
                                                              minibatch_size=self.minibatch_size)

            self.test_label_provider = DataProvider_FitMemory(
                test_y,
                self.signature,
                self.target_dtype,
                verbose=self.verbose,
                minibatch_size=self.minibatch_size)

            start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()
            start_idx, end_idx = self.test_label_provider.get_a_minibatch_idx()
            refresh_theano_shared = True

        if refresh_theano_shared:
            x = test_x
            y = test_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx(self, which=None):
        if which == 'train':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_train()
        elif which == 'valid':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_valid()
        elif which == 'test':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_test()
        else:
            NotImplementedError('which=%s is not supported!!!' % which)

        return start_idx, end_idx, x, y

    def get_dataset(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_dataset()
            y = self.train_label_provider.get_dataset()
        if which == 'valid':
            x = self.valid_input_provider.get_dataset()
            y = self.valid_label_provider.get_dataset()
        if which == 'test':
            x = self.test_input_provider.get_dataset()
            y = self.test_label_provider.get_dataset()

        return x, y

    def get_theano_shared(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_theano_shared()
            y = self.train_label_provider.get_theano_shared()
        if which == 'valid':
            x = self.valid_input_provider.get_theano_shared()
            y = self.valid_label_provider.get_theano_shared()
        if which == 'test':
            x = self.test_input_provider.get_theano_shared()
            y = self.test_label_provider.get_theano_shared()
        return x, y

    def get_minibatch_size(self):
        return self.minibatch_size

    def next_fold(self):
        raise NotImplementedError()


class DataProvider_Scalable(DataProvider):
    # very general worker that takes a dataset and gives out minibatches
    def __init__(self, signature, prefix, train_batch_idx,
                 valid_batch_idx, test_batch_idx,
                 input_dtype, target_dtype, verbose,
                 minibatch_size, c_ordered=True):

        self.prefix = prefix
        self.train_batch_idx = train_batch_idx
        self.valid_batch_idx = valid_batch_idx
        self.test_batch_idx = test_batch_idx
        self.minibatch_size = minibatch_size
        self.signature = signature
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.c_ordered = c_ordered
        self.verbose = verbose
        self._preload_first_batch()

    def next_fold(self):
        self.train_batch_idx += self.valid_batch_idx
        self.valid_batch_idx = self.test_batch_idx
        self._preload_first_batch()
        print
        'Loading dataset for the next fold...Success!'

    def _refresh_trainset(self):
        # print ''
        print
        'Trainset refreshed!'
        self.current_train_batch = 1
        path = self.prefix + str(self.train_batch_idx[0])
        X = RAB_tools.load_pkl(path)
        if self.c_ordered:

            train_x = X['data'].T
        else:
            train_x = X['data']
        train_y = numpy.array(X['labels']).astype('int32')
        signature = X['batch_label']
        # print 'loaded ' + signature

        self.train_input_provider = DataProvider_FitMemory(train_x,
                                                           signature, self.input_dtype,
                                                           minibatch_size=self.minibatch_size,
                                                           verbose=self.verbose
                                                           )
        self.train_label_provider = DataProvider_FitMemory(train_y,
                                                           signature, self.target_dtype,
                                                           verbose=self.verbose,
                                                           minibatch_size=self.minibatch_size)

    def _refresh_validset(self):

        print
        'Validset refreshed!'
        self.current_valid_batch = self.valid_batch_idx[0]
        path = self.prefix + str(self.valid_batch_idx[0])
        X = RAB_tools.load_pkl(path)
        if self.c_ordered:
            valid_x = X['data'].T
        else:
            valid_x = X['data']
        valid_y = numpy.array(X['labels']).astype('int32')
        signature = X['batch_label']
        # print 'loaded ' + signature
        self.valid_input_provider = DataProvider_FitMemory(valid_x,
                                                           signature, self.input_dtype,
                                                           verbose=self.verbose,
                                                           minibatch_size=self.minibatch_size)
        self.valid_label_provider = DataProvider_FitMemory(valid_y,
                                                           signature, self.target_dtype,
                                                           verbose=self.verbose,
                                                           minibatch_size=self.minibatch_size)

    def _refresh_testset(self):

        print
        'Testset refreshed!'
        self.current_test_batch = self.test_batch_idx[0]
        path = self.prefix + str(self.test_batch_idx[0])
        X = RAB_tools.load_pkl(path)
        if self.c_ordered:
            test_x = X['data'].T
        else:
            test_x = X['data']
        test_y = numpy.array(X['labels']).astype('int32')
        signature = X['batch_label']
        # print 'loaded ' + signature
        self.test_input_provider = DataProvider_FitMemory(test_x,
                                                          signature, self.input_dtype,
                                                          verbose=self.verbose,
                                                          minibatch_size=self.minibatch_size)
        self.test_label_provider = DataProvider_FitMemory(test_y,
                                                          signature, self.target_dtype,
                                                          verbose=self.verbose,
                                                          minibatch_size=self.minibatch_size)

    def _preload_first_batch(self):
        print
        'Preloading the trainset and validset'
        self._refresh_trainset()
        self._refresh_validset()
        self._refresh_testset()

    def _refresh(self, which=None):
        if which == 'train':
            self._refresh_trainset()
        elif which == 'valid':
            self._refresh_validset()
        elif which == 'test':
            self._refresh_testset()
        elif which == 'all':
            self._refresh_trainset()
            self._refresh_validset()
            self._refresh_testset()
        else:
            NotImplementedError('which=%s is not supported!' % (which))

    def get_a_minibatch_idx_train(self):
        refresh_theano_shared = False

        start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.train_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            if self.current_train_batch == self.train_batch_idx[-1]:
                # end of one epoch
                print
                '\nreached the end of the training set.'
                start_idx = None
                end_idx = None
                self._refresh(which='train')
            else:
                # print 'loading the next data batch...'
                self.current_train_batch += 1
                path = self.prefix + str(self.current_train_batch)
                X = RAB_tools.load_pkl(path)
                if self.c_ordered:
                    train_x = X['data'].T
                else:
                    train_x = X['data']
                train_y = numpy.array(X['labels']).astype('int32')
                signature = X['batch_label']
                print
                '\nLoaded a new data batch %d' % self.current_train_batch

                self.train_input_provider = DataProvider_FitMemory(train_x,
                                                                   signature,
                                                                   self.input_dtype,
                                                                   verbose=self.verbose,
                                                                   minibatch_size=self.minibatch_size)
                self.train_label_provider = DataProvider_FitMemory(
                    train_y,
                    signature,
                    self.target_dtype,
                    verbose=self.verbose,
                    minibatch_size=self.minibatch_size)

                start_idx, end_idx = self.train_input_provider.get_a_minibatch_idx()
                start_idx, end_idx = self.train_label_provider.get_a_minibatch_idx()
                refresh_theano_shared = True
        if refresh_theano_shared:
            x = train_x
            y = train_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx_valid(self):
        refresh_theano_shared = False
        start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.valid_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            # import ipdb; ipdb.set_trace()
            if self.current_valid_batch == self.valid_batch_idx[-1]:
                # end of one epoch
                print
                '\nreached the end of the validation set.'
                start_idx = None
                end_idx = None
                self._refresh(which='valid')
            else:
                # print 'loading the next data batch...'
                self.current_valid_batch += 1
                path = self.prefix + str(self.current_valid_batch)
                X = RAB_tools.load_pkl(path)
                if self.c_ordered:
                    valid_x = X['data'].T
                else:
                    valid_x = X['data']
                valid_y = numpy.array(X['labels']).astype('int32')
                signature = X['batch_label']
                # print '\nLoaded a new data batch %d'%self.current_valid_batch
                # print '\nloaded ' + signature

                self.valid_input_provider = DataProvider_FitMemory(
                    valid_x,
                    signature,
                    self.input_dtype,
                    verbose=self.verbose,
                    minibatch_size=self.minibatch_size)
                self.valid_label_provider = DataProvider_FitMemory(
                    valid_y,
                    signature,
                    self.target_dtype,
                    verbose=self.verbose,
                    minibatch_size=self.minibatch_size)

                start_idx, end_idx = self.valid_input_provider.get_a_minibatch_idx()
                start_idx, end_idx = self.valid_label_provider.get_a_minibatch_idx()
                refresh_theano_shared = True

        if refresh_theano_shared:
            x = valid_x
            y = valid_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx_test(self):
        refresh_theano_shared = False
        start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()
        start_idx, end_idx = self.test_label_provider.get_a_minibatch_idx()

        if start_idx == None and end_idx == None:
            if self.current_test_batch == self.test_batch_idx[-1]:
                # end of one epoch
                print
                '\nreached the end of the test set.'
                start_idx = None
                end_idx = None
                self._refresh(which='test')
            else:
                # print 'loading the next data batch...'
                self.current_test_batch += 1
                path = self.prefix + str(self.current_test_batch)
                X = RAB_tools.load_pkl(path)
                if self.c_ordered:
                    test_x = X['data'].T
                else:
                    test_x = X['data']
                test_y = numpy.array(X['labels']).astype('int32')
                signature = X['batch_label']
                # print '\nloaded ' + signature
                # print '\nLoaded a new data batch %d'%self.current_test_batch

                self.test_input_provider = DataProvider_FitMemory(
                    test_x,
                    signature,
                    self.input_dtype,
                    verbose=self.verbose,
                    minibatch_size=self.minibatch_size)
                self.test_label_provider = DataProvider_FitMemory(
                    test_y,
                    signature,
                    self.target_dtype,
                    verbose=self.verbose,
                    minibatch_size=self.minibatch_size)

                start_idx, end_idx = self.test_input_provider.get_a_minibatch_idx()
                start_idx, end_idx = self.test_label_provider.get_a_minibatch_idx()
                refresh_theano_shared = True

        if refresh_theano_shared:
            x = test_x
            y = test_y
        else:
            x = None
            y = None

        return start_idx, end_idx, x, y

    def get_a_minibatch_idx(self, which=None):
        if which == 'train':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_train()
        elif which == 'valid':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_valid()
        elif which == 'test':
            start_idx, end_idx, x, y = self.get_a_minibatch_idx_test()
        else:
            NotImplementedError('which=%s is not supported!!!' % which)

        return start_idx, end_idx, x, y

    def get_dataset(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_dataset()
            y = self.train_label_provider.get_dataset()
        if which == 'valid':
            x = self.valid_input_provider.get_dataset()
            y = self.valid_label_provider.get_dataset()
        if which == 'test':
            x = self.test_input_provider.get_dataset()
            y = self.test_label_provider.get_dataset()

        return x, y

    def get_theano_shared(self, which=None):
        if which == 'train':
            x = self.train_input_provider.get_theano_shared()
            y = self.train_label_provider.get_theano_shared()
        if which == 'valid':
            x = self.valid_input_provider.get_theano_shared()
            y = self.valid_label_provider.get_theano_shared()
        if which == 'test':
            x = self.test_input_provider.get_theano_shared()
            y = self.test_label_provider.get_theano_shared()
        return x, y

    def get_minibatch_size(self):
        return self.minibatch_size


class DataProvider_HDF5(object):
    def __init__(self, dataset_name=None, dataset_path=None,
                 minibatch_size=None, n_minibatch_per_chunk=None):
        """

        """
        # first of all, copy the current file into the local temp directory so that
        # clusters do not get choked up by reading remotely from disks
        self.dataset_name = dataset_name

        # temp_folder_path = self._create_temp_folder()
        filename = os.path.basename(use_hdf5)
        # self.hdf5_path_local = os.path.join(temp_folder_path, filename)
        # shutil.copyfile(use_hdf5, self.hdf5_path_local)


        self.minibatch_size = minibatch_size
        self.n_minibatch_per_chunk = n_minibatch_per_chunk
        self.chunk_size = [a * b for a, b
                           in izip(minibatch_size, n_minibatch_per_chunk)]

        self.chunk = [None, None, None]
        self.minibatch_idx_in_chunk = [0, 0, 0]
        self.chunk_idx = [0, 0, 0]
        self.n_chunk = [0, 0, 0]
        # in case the of residual samples after training
        self.last_chunk_size = [0, 0, 0]
        # to handle the last chunk
        self.is_last_chunk[False, False, False]

        self.prepare_dataset()

    def _create_temp_folder(self):
        rval = None
        hostname = socket.gethostname()
        if 0 and hostname == 'ip05':
            # We will use one of the available temp directory on mammouth:
            # $LSCRATCH, $SCRATCH, $HOME, $RAMDISK
            # $LSCRATCH is recommended since it is a temporary directory locally
            # on the computational node, the size is ~100GB and the temp files
            # are automatically deleted at the end of the computation.
            # $SCRATCH is also highly recommended since it has the highest
            # writing/reading throughput (5.5 GB/s globally) and the disk size
            # is also the highest (5 TB per group).
            # Try to avoid $RAMDISK if it has a high throughput (1.2 GB/s)
            # since its size is very small (500 MB). See the documentation at
            # ubi_mm/doc/mammouth_directories.txt for a more exhaustive
            # comparaison between the different available directories.
            temp_dir = os.getenv(self.config['temp_dir'])
            # Temporary file where we store the commands to be launched on
            # the cluster.
            rval = os.path.normpath(os.path.join(temp_dir,
                                                 'temp_history_%s'
                                                 % (miniml.utility.date_time_random_string(time_separator='-'))))
            if os.path.exists(rval):
                raise RuntimeError('Temporary history folder already exists: %s'
                                   % rval)
        else:
            # We are probably running the script on one of the following clusters:
            # briaree, colosse or condor.
            # In that, case we will cerate a temporary folder on the /home directory
            rval = self._create_temp_folder_locally()

        return rval

    def make_dir(self, dir_path):
        if not (os.path.exists(dir_path)):
            try:
                print('... creating directory %s' % dir_path)
                os.mkdir(dir_path)
                # Give the necessary read/write permisssions to the directory.
                os.chmod(dir_path, 0777)
            except Exception, e:
                if e[0] == 17 and 'OSError' in str(type(e)):
                    # This might happen if 2 processes are trying to create
                    # the same directory at the same time.
                    print
                    warnings.warn('Was trying to create the directory %s but it '
                                  'already exists.' % dir_path)
                    return 1
                else:
                    raise
            return 0
        else:
            print
            warnings.warn('The directory %s already exists.' % dir_path)
            return 1

    def _create_temp_folder_locally(self):
        # Get the file path to the temporary folder.
        rval = os.path.join(mkdtemp(), self.signature)
        if os.path.exists(rval):
            raise RuntimeError('Temporary history folder already exists: %s'
                               % rval)
        # Create the temporary folder.
        self.make_dir(rval)
        return rval

    def prepare_dataset(self):
        self.dataset = h5py.File(self.dataset_path, 'r')

        if self.dataset_name == 'mnist':
            self.trainset = self.dataset['trainset']
            self.trainset_label = self.dataset['trainset_label']
            self.validset = self.dataset['validset']
            self.validset_label = self.dataset['validset_label']
            self.testset = self.dataset['testset']
            self.testset_label = self.dataset['testset_label']

            self.n_chunk[0] = self.trainset.shape[0] / self.chunk_size[0]
            self.last_chunk_size[0] = self.chunk_size[0] + self.trainset.shape[0] % self.chunk_size[0]

            self.n_chunk[1] = self.validset.shape[0] / self.chunk_size[1]
            self.last_chunk_size[1] = self.chunk_size[1] + self.validset.shape[0] % self.chunk_size[1]

            self.n_chunk[2] = self.testset.shape[0] / self.chunk_size[2]
            self.last_chunk_size[2] = self.chunk_size[2] + self.testset.shape[0] % self.chunk_size[2]

        elif self.dataset_name == 'imagenet':
            """
            self.dataset.keys() :=
            [u'testset.csr.data', u'testset.csr.indices', u'testset.csr.indptr', u'testset.dense.label',
            u'trainset.csr.data', u'trainset.csr.indices', u'trainset.csr.indptr', u'trainset.dense.label',
            u'validset.csr.data', u'validset.csr.indices', u'validset.csr.indptr', u'validset.dense.label']
            """
            self.trainset['csr.data'] = self.dataset['trainset.csr.data'],
            self.trainset['csr.indices'] = self.dataset['trainset.csr.indices']
            self.trainset['csr.indptr'] = self.dataset['trainset.csr.indptr']
            self.trainset_label = self.dataset['trainset.dense.label']
            self.trainset['shape'] = self.dataset['trainset.shape']

            self.validset['csr.data'] = self.dataset['validset.csr.data'],
            self.validset['csr.indices'] = self.dataset['validset.csr.indices']
            self.validset['csr.indptr'] = self.dataset['validset.csr.indptr']
            self.validset_label = self.dataset['validset.dense.label']
            self.validset['shape'] = self.dataset['validset.shape']

            self.testset['csr.data'] = self.dataset['testset.csr.data'],
            self.testset['csr.indices'] = self.dataset['testset.csr.indices']
            self.testset['csr.indptr'] = self.dataset['testset.csr.indptr']
            self.testset_label = self.dataset['testset.dense.label']
            self.testset['shape'] = self.dataset['testset.shape']

            self.n_chunk[0] = self.trainset['shape'][0] / self.chunk_size[0]
            self.last_chunk_size[0] = self.chunk_size[0] + self.trainset['shape'][0] % self.chunk_size[0]

            self.n_chunk[1] = self.validset['shape'][0] / self.chunk_size[1]
            self.last_chunk_size[1] = self.chunk_size[1] + self.validset['shape'][0] % self.chunk_size[1]

            self.n_chunk[2] = self.testset['shape'][0] / self.chunk_size[2]
            self.last_chunk_size[2] = self.chunk_size[2] + self.testset['shape'][0] % self.chunk_size[2]

        else:
            NotImplementedError(dataset_name + ' is not supported')


class TimeSeriesData(object):
    def __init__(self, data_generator_train, data_generator_valid,
                 data_generator_test, filter_method, chunk_size):
        self.data_generator_train = data_generator_train
        self.data_generator_valid = data_generator_valid
        self.data_generator_test = data_generator_test
        self.filter_method = filter_method
        self.chunk_size = chunk_size

    def get_data_batch(self, which_set):
        """
        this function loads only chunks(batches) inside which minibatches are loaded.
        """
        if which_set == 'train':
            x, y = self.data_generator_train(self.chunk_size)
        elif which_set == 'valid':
            x, y = self.data_generator_valid(self.chunk_size)
        else:
            x, y = self.data_generator_test(self.chunk_size)

        if self.filter_method == 'A':
            x = x.astype('float32')
            y = numpy.argmax(y, axis=1).astype('int32')
        else:
            raise NotImplementedError()
        return x, y


def get_rab_dataset_base_path():
    return os.environ.get('RAB_DATA_PATH')


def get_lisa_dataset_base_path():
    return os.environ.get('LISA_DATA_PATH')


def load_GRO_fun_batches_1008():
    print
    'loading GRO Fun Batches 1008...'
    data_path = get_rab_dataset_base_path() + 'gro_fun/fun_1008_batches/batches_in_20000/'
    if 0:
        batch_idx_train = range(1, 3)
        batch_idx_valid = range(3, 5)
        batch_idx_test = range(5, 7)
    elif 0:
        batch_idx_train = range(1, 13)
        batch_idx_valid = range(13, 15)
        batch_idx_test = range(15, 19)

    else:
        batch_idx_train = range(1, 31)
        batch_idx_valid = range(31, 36)
        batch_idx_test = range(36, 46)

    file_prefix = data_path + 'data_batch_'

    return file_prefix, batch_idx_train, batch_idx_valid, batch_idx_test


def load_face_tubes_batches():
    print
    'loading face tubes batches...'
    data_path = get_rab_dataset_base_path() + 'face_tubes_batches/'
    batch_idx_train = range(1, 4)
    batch_idx_valid = range(4, 7)
    batch_idx_test = range(7, 10)
    file_prefix = data_path + 'data_batch_'
    return file_prefix, batch_idx_train, batch_idx_valid, batch_idx_test


def load_GRO_fun_batches():
    print
    'loading GRO Fun Batches...'
    data_path = get_rab_dataset_base_path() + 'gro_fun/fun_744_batches/batches_in_20000/'
    if 0:
        batch_idx_train = range(1, 3)
        batch_idx_valid = range(3, 5)
        batch_idx_test = range(5, 7)
    elif 0:
        batch_idx_train = range(1, 13)
        batch_idx_valid = range(13, 15)
        batch_idx_test = range(15, 19)

    else:
        batch_idx_train = range(1, 31)
        batch_idx_valid = range(31, 36)
        batch_idx_test = range(36, 46)

    file_prefix = data_path + 'data_batch_'

    return file_prefix, batch_idx_train, batch_idx_valid, batch_idx_test


def save_gro_fun_just_extra_attrs():
    base_path = get_rab_dataset_base_path()
    path = base_path + 'gro_fun/fun_744_labeled_split_70_15_15/'
    train_x_744 = numpy.load(path + 'train_x.npy')
    train_y_744 = numpy.load(path + 'train_y.npy')
    valid_x_744 = numpy.load(path + 'valid_x.npy')
    test_x_744 = numpy.load(path + 'test_x.npy')

    path = base_path + 'gro_fun/fun_1008_labeled_split_70_15_15/'
    train_x_1008 = numpy.load(path + 'train_x.npy')
    train_y_1008 = numpy.load(path + 'train_y.npy')
    valid_x_1008 = numpy.load(path + 'valid_x.npy')
    test_x_1008 = numpy.load(path + 'test_x.npy')
    import ipdb;
    ipdb.set_trace()

    old_attributes = []
    for i in range(test_x_744.shape[1]):
        print
        i
        a = test_x_744[:, i]
        if numpy.sum(a) == 0:
            continue
        found = False
        for j in range(test_x_1008.shape[1]):
            b = test_x_1008[:, j]
            sign = numpy.sum(a - b) == 0
            if sign:
                old_attributes.append(j)
                found = True
                break

        if not found:
            print
            'not found'

    import ipdb;
    ipdb.set_trace()


def load_SVNH():
    pass


def load_temp():
    print
    'loading TEMP dataset...'
    base_path = get_rab_dataset_base_path()
    train_x = numpy.load(base_path + 'temp/train_x.npy')
    train_y = numpy.load(base_path + 'temp/train_y.npy')
    valid_x = numpy.load(base_path + 'temp/valid_x.npy')
    valid_y = numpy.load(base_path + 'temp/valid_y.npy')
    test_x = valid_x
    test_y = valid_y
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_iris():
    print
    'loading Iris dataset...'
    iris = datasets.load_iris()
    inputs = iris.data
    outputs = iris.target
    idx = range(150)
    numpy.random.seed(1)
    numpy.random.shuffle(idx)

    inputs = inputs[idx, :]
    outputs = outputs[idx]
    train = inputs[0:100, :].astype('float32')
    train_label = outputs[0:100].astype('int32')
    valid = inputs[100:125, :].astype('float32')
    valid_label = outputs[100:125].astype('int32')
    test = inputs[125:150, :].astype('float32')
    test_label = outputs[125:150].astype('int32')

    return train, train_label, valid, valid_label, test, test_label


def load_diabetes():
    print
    'loading diabetes regression dataset...'
    diabetes = datasets.load_diabetes()
    inputs = diabetes.data
    outputs = diabetes.target
    idx1 = 221
    idx2 = 331
    idx3 = 442
    idx = range(442)
    numpy.random.seed(1)
    numpy.random.shuffle(idx)

    inputs = inputs[idx, :]
    outputs = outputs[idx]
    train = inputs[0:idx1, :].astype('float32')
    train_label = outputs[0:idx1].astype('int32')
    valid = inputs[idx1:idx2, :].astype('float32')
    valid_label = outputs[idx1:idx2].astype('int32')
    test = inputs[idx2:idx3, :].astype('float32')
    test_label = outputs[idx2:idx3].astype('int32')

    return train, train_label, valid, valid_label, test, test_label


def load_TFD(style='unsupervised', normalize=False):
    print
    'loading TFD...'
    # 98058
    unlabeled = tfd.TFD(which_set='unlabeled', center=False, scale=False)
    # 2913
    train = tfd.TFD(which_set='train', center=False, scale=False)
    # 428
    valid = tfd.TFD(which_set='valid', center=False, scale=False)
    # 837
    test = tfd.TFD(which_set='test', center=False, scale=False)
    # 3341
    full_train = tfd.TFD(which_set='full_train', center=False, scale=False)

    full_train_x = full_train.X
    full_train_y = numpy.zeros((full_train_x.shape[0],))
    unlabeled_x = unlabeled.X
    unlabeled_y = numpy.zeros((unlabeled_x.shape[0],))
    train_x = train.X
    train_y = train.y
    valid_x = valid.X
    valid_y = valid.y
    test_x = test.X
    test_y = test.y
    print
    'full train size: ', full_train_x.shape
    print
    'unlabeled size: ', unlabeled_x.shape
    print
    'trainset size: ', train_x.shape
    print
    'validset size: ', valid_x.shape
    print
    'testset size: ', test_x.shape

    if style == 'unsupervised':
        print
        'TFD unsupervised loaded. unlabeled as train, valid as valid, test as test.'
        train_x_f = unlabeled_x.astype('float32')
        train_y_f = unlabeled_y.astype('int32')
        valid_x_f = valid_x.astype('float32')
        valid_y_f = valid_y.astype('int32')
        test_x_f = test_x.astype('float32')
        test_y_f = test_y.astype('int32')

    elif style == 'all_unsupervised':
        X = numpy.concatenate((unlabeled.X, full_train.X), axis=0)
        X = RAB_tools.center_pixels(X).astype('float32')
        idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(X.shape[0])
        train_x_f = X[idx1]
        valid_x_f = X[idx2]
        test_x_f = X[idx3]
        hdf5_path = get_rab_dataset_base_path() + 'tfd_centered_scaled.hdf5'
        RAB_tools.pkl_to_hdf5(train_x_f, valid_x_f, test_x_f, hdf5_path)

    elif style == 'unsupervised_and_supervised':
        print
        'TFD is loaded such that the train_x is the original unsupervised, valid' \
        ' and test is the same orginial supervised part.'
        train_x_f = unlabeled_x.astype('float32')
        train_y_f = unlabeled_y.astype('int32')
        valid_x_f = numpy.concatenate((train_x, valid_x, test_x))
        valid_y_f = numpy.concatenate((train_y, valid_y, test_y))
        test_x_f = valid_x_f
        test_y_f = valid_y_f
    else:
        print
        'using TFD supervised...'
        train_x_f = train_x.astype('float32')
        train_y_f = train_y.astype('int32').flatten()
        valid_x_f = valid_x.astype('float32')
        valid_y_f = valid_y.astype('int32').flatten()
        test_x_f = test_x.astype('float32')
        test_y_f = test_y.astype('int32').flatten()

        print
        'train size', train_x_f.shape
        print
        'valid size', valid_x_f.shape
        print
        'test size', test_x_f.shape

    import ipdb;
    ipdb.set_trace()

    return train_x_f, train_y_f, valid_x_f, valid_y_f, test_x_f, test_y_f


def load_GRO_fun_unlabeled(which='744'):
    if which == '744':
        base_path = None
    elif which == '1008':
        base_path = None
    print
    'loading data for prediction gro fun 1008...'

    train_x_path = base_path + 'train_x_labeled.npy'
    train_y_path = base_path + 'train_y_labeled.npy'
    test_x_path = base_path + 'test_x_labeled.npy'
    test_y_path = base_path + 'test_y_labeled.npy'
    if os.path.exists(train_x_path):
        train_x_all = numpy.load(base_path + 'train_x_labeled.npy')
        train_y_all = numpy.load(base_path + 'train_y_labeled.npy')
    else:
        raise Exception('s% not found' % train_x_path)
    if os.path.exists(test_x_path):
        test_x_all = numpy.load(base_path + 'test_x_labeled.npy')
        test_y_all = numpy.load(base_path + 'test_y_labeled.npy')
    else:
        raise Exception('s% not found' % test_x_path)

    if 1:
        train_idx = slice(0, 600000)
        valid_idx = slice(600000, 686044)
        train_x = train_x_all[idx[train_idx]]
        train_y = train_y_all[idx[train_idx]]
        valid_x = train_x_all[idx[valid_idx]]
        valid_y = train_y_all[idx[valid_idx]]

        test_x = test_x_all
        test_y = test_y_all

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_facetube_singles():
    # base_path = '/data/lisatmp/desjagui/data/emotiwfaces/gcn_whitened/'
    base_path = get_rab_dataset_base_path() + 'facetube_pairs_whitened/'

    train_x = numpy.load(base_path + 'train_x.npy')
    train_y = numpy.load(base_path + 'train_y.npy')
    valid_x = numpy.load(base_path + 'valid_x.npy')
    valid_y = numpy.load(base_path + 'valid_y.npy')

    import ipdb;
    ipdb.set_trace()
    return


def load_GRO_fun_unlabeled():
    pass


def load_GRO_fun_labeled(mode='normal', which='744', small=False,
                         use_differential_targets=False, directory='lisatmp'):
    """
    883948 training examples
    """
    if directory == 'lisatmp2':
        base_path = get_rab_dataset_base_path() + 'gro_fun/'
    else:
        base_path = '/data/lisatmp/yaoli/datasets/gro_fun/'

    if which == '744':
        print
        'loading gro fun 744 labeled...'
        path = base_path + 'fun_744_labeled_split_70_15_15/'

    elif which == '1008':
        print
        'loading gro fun 1008 labeled...'
        path = base_path + 'fun_1008_labeled_split_70_15_15/'
    X = {}
    # import pdb; pdb.set_trace()
    X['train_x'] = numpy.load(path + 'train_x.npy')
    X['train_y'] = numpy.load(path + 'train_y_probability.npy')
    X['valid_x'] = numpy.load(path + 'valid_x.npy')
    X['valid_y'] = numpy.load(path + 'valid_y_probability.npy')
    X['test_x'] = numpy.load(path + 'test_x.npy')
    X['test_y'] = numpy.load(path + 'test_y_probability.npy')

    if use_differential_targets:

        # print 'composing differential fun targets...'
        # biases = numpy.load(os.environ.get('GRO') + 'simplified_pipeline/user_specific_biases/p_y_given_x_infos_fold_0.npz')
        # train_bias = biases['p_y_given_x_train'].astype('float32')
        # valid_bias = biases['p_y_given_x_valid'].astype('float32')
        # test_bias = biases['p_y_given_x_test'].astype('float32')
        # X['train_y'] = X['train_y'] - train_bias
        # X['valid_y'] = X['valid_y'] - valid_bias
        # X['test_y'] = X['test_y'] - test_bias

        # numpy.save(path+'train_y_without_player_bias.npy', X['train_y'])
        # numpy.save(path+'valid_y_without_player_bias.npy', X['valid_y'])
        # numpy.save(path+'test_y_without_player_bias.npy', X['test_y'])
        print
        'loading labels without player bias... '
        X['train_y'] = numpy.load(path + 'train_y_without_player_bias.npy')
        X['valid_y'] = numpy.load(path + 'valid_y_without_player_bias.npy')
        X['test_y'] = numpy.load(path + 'test_y_without_player_bias.npy')

        if 0:
            print
            'using artificial targets...'
            targets = numpy.load(path + 'train_y_artificial_differential.npy')
            X['train_y'] = targets

    # load pids for train, valid, test
    train_pid = numpy.load(path + 'train_pid.npy').astype('float32')
    valid_pid = numpy.load(path + 'valid_pid.npy').astype('float32')
    test_pid = numpy.load(path + 'test_pid.npy').astype('float32')
    train_pid = train_pid.reshape((train_pid.shape[0], 1))
    valid_pid = valid_pid.reshape((valid_pid.shape[0], 1))
    test_pid = test_pid.reshape((test_pid.shape[0], 1))

    # tr = set(train_pid.flatten().tolist())
    # te = set(test_pid.flatten().tolist())
    # counter = 0
    # for id in te-tr:
    #    counter += numpy.sum(test_pid==id)
    # print counter/(test_pid.shape[0]+.0)
    # import ipdb; ipdb.set_trace()
    # combine pids with attrs

    print
    'combining pids with attributes, pid appends to attrs...'
    X['train_x'] = numpy.concatenate((X['train_x'], train_pid), axis=1)
    X['valid_x'] = numpy.concatenate((X['valid_x'], valid_pid), axis=1)
    X['test_x'] = numpy.concatenate((X['test_x'], test_pid), axis=1)

    if mode == 'reverse':
        r_train_x = X['valid_x']
        r_train_y = X['valid_y']
        r_valid_x = X['train_x']
        r_valid_y = X['train_y']
        r_test_x = X['test_x']
        r_test_y = X['test_y']

    elif mode == 'normal':
        r_train_x = X['train_x']
        r_train_y = X['train_y']
        r_valid_x = X['valid_x']
        r_valid_y = X['valid_y']
        r_test_x = X['test_x']
        r_test_y = X['test_y']
    else:
        raise NotImplementedError()

    if small:
        # separate a validset from trainset
        r_train_x = train_x[0:60000]
        r_train_y = train_y[0:60000]
        r_valid_x = train_x[60000:90000]
        r_valid_y = train_y[60000:90000]
        r_test_x = test_x[0:30000]
        r_test_y = test_y[0:30000]
    # import ipdb; ipdb.set_trace()
    return r_train_x, r_train_y, r_valid_x, r_valid_y, r_test_x, r_test_y


def load_GRO_kill_ratio(mode='normal', which='620'):
    print
    'loading gro kill ratio...'
    if which == '620':
        base_path = get_rab_dataset_base_path() + 'gro_kill/kill_ratio_620_labeled_split_70_15_15/'
    elif which == '840':
        base_path = get_rab_dataset_base_path() + 'gro_kill/kill_ratio_1008_labeled_split_70_15_15/'

    train_x = numpy.load(base_path + 'train_x.npy').astype('float32')
    train_y = numpy.load(base_path + 'train_y.npy').astype('float32')
    valid_x = numpy.load(base_path + 'valid_x.npy').astype('float32')
    valid_y = numpy.load(base_path + 'valid_y.npy').astype('float32')
    test_x = numpy.load(base_path + 'test_x.npy').astype('float32')
    test_y = numpy.load(base_path + 'test_y.npy').astype('float32')

    if mode == 'normal':
        r_train_x = train_x
        r_train_y = train_y
        r_valid_x = valid_x
        r_valid_y = valid_y
        r_test_x = test_x
        r_test_y = test_y
    elif model == 'reverse':
        r_train_x = valid_x
        r_train_y = valid_y
        r_valid_x = train_x
        r_valid_y = train_y
        r_test_x = test_x
        r_test_y = test_y

    return r_train_x, r_train_y, r_valid_x, r_valid_y, r_test_x, r_test_y


def preprocess_GRO_kill_ratio_1008():
    base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_1008_labeled_and_unlabeled/'
    print
    'making train_x and train_y'
    train_x_part1 = numpy.load(base_path + 'trainset_attrs_part1.npy').astype('float32')
    train_x_part2 = numpy.load(base_path + 'trainset_attrs_part2.npy').astype('float32')
    train_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_kill_ratios.pkl')).astype('float32')

    train_x_all = numpy.concatenate((train_x_part1, train_x_part2))
    n_minibatch, n_players, attr = train_x_all.shape
    train_x_all = train_x_all.reshape((n_minibatch * 30, 16, attr))
    train_x_all = train_x_all[:, 0, 168:]
    train_y_all = train_y_all.reshape((n_minibatch * 30,))

    test_x_all = numpy.load(base_path + 'testset_attrs.npy').astype('float32')
    test_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_kill_ratios.pkl')).astype('float32')

    n_minibatch, n_players, attr = test_x_all.shape
    test_x_all = test_x_all.reshape((n_minibatch, 16, attr))
    test_x_all = test_x_all[:, 0, 168:]
    test_y_all = test_y_all.reshape((n_minibatch,))

    X = numpy.concatenate((train_x_all, test_x_all))
    Y = numpy.concatenate((train_y_all, test_y_all))
    effective = Y != -999
    X = X[effective]
    Y = numpy.abs(Y[effective]).astype('float32')

    train_idx, valid_idx, test_idx = RAB_tools.divide_to_3_folds(X.shape[0],
                                                                 mode=[.70, .15, .15])
    train_x = X[train_idx]
    train_y = Y[train_idx]
    valid_x = X[valid_idx]
    valid_y = Y[valid_idx]
    test_x = X[test_idx]
    test_y = Y[test_idx]

    print
    'saving...'

    save_path = get_rab_dataset_base_path() + 'gro_kill/kill_ratio_840_labeled_split_70_15_15/'
    numpy.save(save_path + 'train_x.npy', train_x)
    numpy.save(save_path + 'train_y.npy', train_y)
    numpy.save(save_path + 'valid_x.npy', valid_x)
    numpy.save(save_path + 'valid_y.npy', valid_y)
    numpy.save(save_path + 'test_x.npy', test_x)
    numpy.save(save_path + 'test_y.npy', test_y)
    numpy.savez(save_path + 'splitting_indices.npz',
                train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)


def preprocess_GRO_kill_ratio():
    base_path = get_rab_dataset_base_path() + 'gro_win_and_kill_ratio/'
    train_x_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_X.pkl'))
    train_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_y_kill_ratios.pkl'))

    test_x = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_X.pkl'))
    test_y = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_y_kill_ratios.pkl'))

    # get rid of -999 and do absolute value
    X = numpy.concatenate((train_x_all, test_x))
    Y = numpy.concatenate((train_y_all, test_y))
    effective = Y != -999
    X = X[effective]
    Y = numpy.abs(Y[effective])

    if 1:
        train_idx, valid_idx, test_idx = RAB_tools.divide_to_3_folds(X.shape[0],
                                                                     mode=[.70, .15, .15])
        train_x = X[train_idx]
        train_y = Y[train_idx]
        valid_x = X[valid_idx]
        valid_y = Y[valid_idx]
        test_x = X[test_idx]
        test_y = Y[test_idx]

    save_path = get_rab_dataset_base_path() + 'gro_kill/kill_ratio_620_labeled_split_70_15_15/'
    import ipdb;
    ipdb.set_trace()
    numpy.save(save_path + 'train_x.npy', train_x)
    numpy.save(save_path + 'train_y.npy', train_y)
    numpy.save(save_path + 'valid_x.npy', valid_x)
    numpy.save(save_path + 'valid_y.npy', valid_y)
    numpy.save(save_path + 'test_x.npy', test_x)
    numpy.save(save_path + 'test_y.npy', test_y)


def load_GRO_winner():
    print
    "loading GRO winner dataset"
    # X_path = "/data/lisa/data/ubi/gro/simulator/ob_partial_balance_436344_X.pkl"
    # Y_path = "/data/lisa/data/ubi/gro/simulator/ob_partial_balance_436344_Y.pkl"
    # X_path = '/mnt/scratch/bengio/yaoli001/data/ob_partial_balance_436344_X.pkl'
    # Y_path = '/mnt/scratch/bengio/yaoli001/data/ob_partial_balance_436344_Y.pkl'
    base_path = get_rab_dataset_base_path() + 'gro_win_and_kill_ratio/'

    train_x_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_X.pkl'))
    train_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_y_winners.pkl'))

    test_x = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_X.pkl'))
    test_y = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_y_winners.pkl'))

    # get rid of the last 10 invalid train
    train_x_all = train_x_all[:327470]
    train_y_all = train_y_all[:327470]

    # there are 436344(327470 for train, 108872 for test) in total

    # converting labels
    print
    'converting labels'
    train_y_all[train_y_all == 0] = 0
    train_y_all[train_y_all == 1] = 1
    train_y_all[train_y_all == 0.5] = 2

    test_y[test_y == 0] = 0
    test_y[test_y == 1] = 1
    test_y[test_y == 0.5] = 2

    if 1:
        print
        'splitting into train, valid and test'
        id1 = 200000
        # id1 = 20000
        train_x = train_x_all[0:id1, :].astype('float32')
        train_y = train_y_all[0:id1].astype('int32')
        valid_x = train_x_all[id1:, :].astype('float32')
        valid_y = train_y_all[id1:].astype('int32')
        test_x = test_x.astype('float32')
        test_y = test_y.astype('int32')
    else:
        # for test
        id1 = 5000
        id2 = 6000
        id3 = 7000
        train = inputs[0:id1, :].astype('float32')
        train_label = outputs[0:id1].astype('int32')
        valid = inputs[id1:id2, :].astype('float32')
        valid_label = outputs[id1:id2].astype('int32')
        test = inputs[id2:id3, :].astype('float32')
        test_label = outputs[id2:id3].astype('int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_GRO_kill_ratio_calibration():
    print
    'loading gro kill ratio calibration...'
    # path = '/u/yaoli/GRO_NEW/ubi_mm/simplified_pipeline/archive/best_model_params/jobman_random_forest/fold_info.pkl'
    path = '/u/yaoli/GRO_NEW/ubi_mm/simplified_pipeline/archive/best_model_params/jobman_maxout/fold_info.npz'
    x = numpy.load(path)['p_y_given_x_test']
    base_path = get_rab_dataset_base_path() + 'gro_kill/kill_ratio_620_labeled_split_70_15_15/'
    y = numpy.load(base_path + 'test_y.npy').astype('float32')

    train_x = numpy.array([x, x ** 2, x ** 3]).astype('float32').T
    train_y = y
    valid_x = train_x
    valid_y = train_y
    test_x = train_x
    test_y = train_y
    # import ipdb; ipdb.set_trace()
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_cifar10_raw():
    print
    'loading cifar10..., no validation set, use testset as validset'
    train, test = cifar10_wrapper.get_cifar10_raw()
    trainset = train.X
    trainset_label = train.get_targets().astype('int32')
    testset = test.X
    testset_label = test.get_targets().astype('int32')
    validset = testset
    validset_label = testset_label
    return (trainset, trainset_label, validset,
            validset_label, testset, testset_label)


def load_test_mix():
    rng_numpy, _ = RAB_tools.get_two_rngs()
    n_row = 200
    n_col = 144
    gaussian_1 = numpy.asarray(rng_numpy.normal(loc=0, scale=.5,
                                                size=(n_row, n_col)), dtype='float32')
    gaussian_2 = numpy.asarray(rng_numpy.normal(loc=1, scale=.5,
                                                size=(n_row, n_col)), dtype='float32')
    # gaussian_2 = gaussian_1
    train_x = numpy.concatenate([gaussian_1, gaussian_2])
    train_y = numpy.concatenate([numpy.zeros((n_row,)), numpy.ones((n_row,))]).astype('int32')

    idx = RAB_tools.shuffle_idx(train_x.shape[0])
    train_x = train_x[idx]
    train_y = train_y[idx]

    test_x = gaussian_2
    test_y = numpy.zeros((n_row,)).astype('int32')

    return train_x, train_y, test_x, test_y


def load_stackexchange(small=False):
    base_path = get_rab_dataset_base_path() + 'stackexchange/'
    if small:
        print
        'loading stackexchange small'
        x = RAB_tools.load_pkl(base_path + 'train_x_small_60k.pkl').astype('float32')
        y = RAB_tools.load_pkl(base_path + 'train_y_small_60k.pkl').astype('float32')
        idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(x.shape[0])
        train_x = x[idx1]
        train_y = y[idx1]
        valid_x = x[idx2]
        valid_y = y[idx2]
        test_x = x[idx1]
        test_y = y[idx2]

        return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_biodesix(task, unsupervised=False, mix=False, pretrain=False):
    assert task == [0] or task == [1] or task == [2] or task == [3] or task == [4] or task == [5] or task == [0,
                                                                                                              1] or task == [
        0, 1, 2, 3, 4, 5] or task == [1, 2, 3, 4, 5]
    base_path = get_rab_dataset_base_path() + 'biodesix/'

    def load_a_task(i):
        assert i in [0, 1, 2, 3, 4, 5]
        print
        'loading biodesix p%d' % i
        if i == 0:
            p0_train_x = numpy.load(base_path +
                                    'pure_unsupervised_144.npy').astype('float32')
            p0_train_y = (numpy.zeros((p0_train_x.shape[0], 2))).astype('int32')

            p0_train_y[:, 0] += -1

            return p0_train_x, p0_train_y

        else:
            train_x = numpy.load(base_path +
                                 'pure_supervised_p%d_train_x_144.npy' % i).astype('float32')
            train_y = numpy.load(base_path +
                                 'pure_supervised_p%d_train_y_144.npy' % i).astype('int32')
            test_x = numpy.load(base_path +
                                'pure_supervised_p%d_test_x_144.npy' % i).astype('float32')
            test_y = numpy.load(base_path +
                                'pure_supervised_p%d_test_y_144.npy' % i).astype('int32')

            train_y = train_y[:, numpy.newaxis]
            test_y = test_y[:, numpy.newaxis]
            extra_train_y = numpy.zeros((train_y.shape[0], 1)) + i
            extra_test_y = numpy.zeros((test_y.shape[0], 1)) + i
            train_y = numpy.concatenate([train_y, extra_train_y], axis=1).astype('int32')
            test_y = numpy.concatenate([test_y, extra_test_y], axis=1).astype('int32')
            return train_x, train_y, test_x, test_y

    if task == [1]:
        train_x, train_y, test_x, test_y = load_a_task(1)
    elif task == [2]:
        train_x, train_y, test_x, test_y = load_a_task(2)
    elif task == [3]:
        train_x, train_y, test_x, test_y = load_a_task(3)
    elif task == [4]:
        train_x, train_y, test_x, test_y = load_a_task(4)
    elif task == [5]:
        train_x, train_y, test_x, test_y = load_a_task(5)

    elif task == [0]:
        p0_train_x, p0_train_y = load_a_task(0)

    elif task == [0, 1]:
        p0_train_x, p0_train_y = load_a_task(0)
        p1_train_x, p1_train_y, p1_test_x, p1_test_y = load_a_task(1)
        print
        'combing p0 and p1...'
        train_x = numpy.concatenate([p0_train_x, p1_train_x], axis=0)
        train_y = numpy.concatenate([p0_train_y, p1_train_y], axis=0)

        test_x = p1_test_x
        test_y = p1_test_y

    elif pretrain:
        print
        'loading all train data, sup+unsup, for pretraining'
        p0_train_x, p0_train_y = load_a_task(0)
        p1_train_x, p1_train_y, p1_test_x, p1_test_y = load_a_task(1)
        p2_train_x, p2_train_y, p2_test_x, p2_test_y = load_a_task(2)
        p3_train_x, p3_train_y, p3_test_x, p3_test_y = load_a_task(3)
        p4_train_x, p4_train_y, p4_test_x, p4_test_y = load_a_task(4)
        p5_train_x, p5_train_y, p5_test_x, p5_test_y = load_a_task(5)

        x = [p0_train_x, p1_train_x, p2_train_x, p3_train_x, p4_train_x, p5_train_x
             ]
        x = numpy.concatenate(x, axis=0).astype('float32')
        y = numpy.zeros((x.shape[0],)).astype('int32')
        x = RAB_tools.zero_mean_unit_variance(x)
        train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                            test_size=0.20, random_state=1234)
        return train_x, train_y, test_x, test_y

    elif task == [0, 1, 2, 3, 4, 5]:
        p0_train_x, p0_train_y = load_a_task(0)
        p1_train_x, p1_train_y, p1_test_x, p1_test_y = load_a_task(1)
        p2_train_x, p2_train_y, p2_test_x, p2_test_y = load_a_task(2)
        p3_train_x, p3_train_y, p3_test_x, p3_test_y = load_a_task(3)
        p4_train_x, p4_train_y, p4_test_x, p4_test_y = load_a_task(4)
        p5_train_x, p5_train_y, p5_test_x, p5_test_y = load_a_task(5)

        sup_size = p1_train_x.shape[0] + p2_train_x.shape[0] + \
                   p3_train_x.shape[0] + p4_train_x.shape[0] + \
                   p5_train_x.shape[0]
        unsup_size = p0_train_x.shape[0]

        '''
        # rebalance dataset
        # NOTE: do not rebalance dataset since it is not compatible with KFold CV!!!
        # The rebalancing should be done inside the data_providers
        print 'rebalance dataset'
        k = unsup_size / sup_size
        p1_train_x, p1_train_y = RAB_tools.duplicate_dataset(p1_train_x, p1_train_y, k)
        p2_train_x, p2_train_y = RAB_tools.duplicate_dataset(p2_train_x, p2_train_y, k)
        p3_train_x, p3_train_y = RAB_tools.duplicate_dataset(p3_train_x, p3_train_y, k)
        p4_train_x, p4_train_y = RAB_tools.duplicate_dataset(p4_train_x, p4_train_y, k)
        p5_train_x, p5_train_y = RAB_tools.duplicate_dataset(p5_train_x, p5_train_y, k)
        '''

        train_x = [p0_train_x, p1_train_x, p2_train_x, p3_train_x, p4_train_x, p5_train_x]
        train_y = [p0_train_y, p1_train_y, p2_train_y, p3_train_y, p4_train_y, p5_train_y]
        test_x = [p1_test_x, p2_test_x, p3_test_x, p4_test_x, p5_test_x]
        test_y = [p1_test_y, p2_test_y, p3_test_y, p4_test_y, p5_test_y]
        train_x = numpy.concatenate(train_x, axis=0)
        train_y = numpy.concatenate(train_y, axis=0)
        test_x = numpy.concatenate(test_x, axis=0)
        test_y = numpy.concatenate(test_y, axis=0)

        split_idx = train_x.shape[0]
        data = numpy.concatenate([train_x, test_x], axis=0)
        data = RAB_tools.zero_mean_unit_variance(data)
        # data = RAB_tools.compute_rank(data)
        # data = RAB_tools.uniformization(data)
        train_x = data[:split_idx]
        test_x = data[split_idx:]
        idx = RAB_tools.shuffle_idx(train_x.shape[0])
        train_x = train_x[idx]
        train_y = train_y[idx]

        return train_x, train_y, test_x, test_y

    elif task == [1, 2, 3, 4, 5]:
        p1_train_x, p1_train_y, p1_test_x, p1_test_y = load_a_task(1)
        p2_train_x, p2_train_y, p2_test_x, p2_test_y = load_a_task(2)
        p3_train_x, p3_train_y, p3_test_x, p3_test_y = load_a_task(3)
        p4_train_x, p4_train_y, p4_test_x, p4_test_y = load_a_task(4)
        p5_train_x, p5_train_y, p5_test_x, p5_test_y = load_a_task(5)

        train_x = [p1_train_x, p2_train_x, p3_train_x, p4_train_x, p5_train_x]
        train_y = [p1_train_y, p2_train_y, p3_train_y, p4_train_y, p5_train_y]
        test_x = [p1_test_x, p2_test_x, p3_test_x, p4_test_x, p5_test_x]
        test_y = [p1_test_y, p2_test_y, p3_test_y, p4_test_y, p5_test_y]
        train_x = numpy.concatenate(train_x, axis=0)
        train_y = numpy.concatenate(train_y, axis=0)
        test_x = numpy.concatenate(test_x, axis=0)
        test_y = numpy.concatenate(test_y, axis=0)

        split_idx = train_x.shape[0]
        data = numpy.concatenate([train_x, test_x], axis=0)
        data = RAB_tools.zero_mean_unit_variance(data)
        train_x = data[:split_idx]
        test_x = data[split_idx:]
        idx = RAB_tools.shuffle_idx(train_x.shape[0])
        train_x = train_x[idx]
        train_y = train_y[idx]

        # clean up labels
        train_y = train_y[:, 0]
        test_y = test_y[:, 0]

        return train_x, train_y, test_x, test_y

    else:
        raise NotImplementedError()

    if not mix:

        # do not mix train with test
        # preprocessing
        split_idx = train_x.shape[0]
        data = numpy.concatenate([train_x, test_x], axis=0)
        data = RAB_tools.zero_mean_unit_variance(data)
        # data = RAB_tools.compute_rank(data)

        train_x = data[:split_idx]
        test_x = data[split_idx:]
        idx = RAB_tools.shuffle_idx(train_x.shape[0])
        train_x = train_x[idx]
        train_y = train_y[idx]

        # clean up labels
        train_y = train_y[:, 0]
        test_y = test_y[:, 0]
    else:
        # train_x, train_y, test_x, test_y = load_test_mix()
        print
        'mixing train with test...'
        '''
        train_x = numpy.concatenate([train_x, test_x], axis=0)
        train_y = numpy.zeros((78,)).astype('int32')
        test_y = numpy.ones((78,)).astype('int32')
        train_y = numpy.concatenate([train_y, test_y], axis=0)
        
        '''
        # mix train with test
        train_y = numpy.zeros(train_y.shape)[:, 0].astype('int32')
        test_y = numpy.ones(test_y.shape)[:, 0].astype('int32')
        train_y = numpy.concatenate([train_y, test_y], axis=0)

        idx = train_x.shape[0]

        train_x = numpy.vstack([train_x, test_x])
        train_x = RAB_tools.zero_mean_unit_variance(train_x)
        test_x = train_x[idx:]

        idx = RAB_tools.shuffle_idx(train_x.shape[0])
        train_x = train_x[idx]
        train_y = train_y[idx]

    return train_x, train_y, test_x, test_y


def load_sin():
    numpy.random.RandomState(1234)

    k = 1000.
    x = numpy.arange(k)
    noise = numpy.random.rand(k) - 0.5 - k
    y = numpy.sin(2 * numpy.pi * x / k) + noise
    pylab.plot(y, x, 'o')
    pylab.show()
    import ipdb;
    ipdb.set_trace()


def load_mnist_original(which_set):
    path = "${PYLEARN2_DATA_PATH}/mnist/"
    if which_set == 'train':
        im_path = path + 'train-images-idx3-ubyte'
        label_path = path + 'train-labels-idx1-ubyte'
    else:
        assert which_set == 'test'
        im_path = path + 't10k-images-idx3-ubyte'
        label_path = path + 't10k-labels-idx1-ubyte'
        # Path substitution done here in order to make the lower-level
        # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
        # the Deep Learning Tutorials, or in another package).
    im_path = serial.preprocess(im_path)
    label_path = serial.preprocess(label_path)
    topo_view = read_mnist_images(im_path)
    y = read_mnist_labels(label_path)
    return topo_view, y


def load_mnist_scaled(binarize=True):
    print
    'load MNIST 14 by 14, binarized'
    base_path = get_rab_dataset_base_path() + 'mnist_14_by_14/'
    train_x = numpy.load(base_path + 'train_x.npy')
    train_y = numpy.load(base_path + 'train_y.npy')
    valid_x = numpy.load(base_path + 'valid_x.npy')
    valid_y = numpy.load(base_path + 'valid_y.npy')
    test_x = numpy.load(base_path + 'test_x.npy')
    test_y = numpy.load(base_path + 'test_y.npy')

    if binarize:
        train_x = (train_x > 0.5).astype('float32')
        valid_x = (valid_x > 0.5).astype('float32')
        test_x = (test_x > 0.5).astype('float32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_debug_dataset():
    print
    'load debug dataset'
    train_x = (numpy.zeros((100, 4)) + [0, 1, 2, 3]).astype('float32')
    train_y = numpy.asarray([0] * 100).astype('int32')
    valid_x = train_x
    valid_y = train_y
    test_x = train_x
    test_y = train_y

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def preprocess_mnist_scaled(shuffle=True):
    print
    'load original mnist and scaled each example to 14*14'
    tr_x, tr_y = load_mnist_original(which_set='train')
    te_x, te_y = load_mnist_original(which_set='test')
    tr_x = (RAB_tools.resize_img(tr_x) / 255.).astype('float32')
    te_x = (RAB_tools.resize_img(te_x) / 255.).astype('float32')
    tr_y = tr_y.astype('int32')
    te_y = te_y.astype('int32')
    if shuffle:
        idx = range(tr_x.shape[0])
        print
        'shuffling the trainset...'
        numpy.random.RandomState(1234)
        numpy.random.shuffle(idx)
        idx_train = idx[:50000]
        idx_valid = idx[50000:]
    train_x = tr_x[idx_train]
    train_y = tr_y[idx_train]
    valid_x = tr_x[idx_valid]
    valid_y = tr_y[idx_valid]
    test_x = te_x
    test_y = te_y
    base_path = get_rab_dataset_base_path() + 'mnist_14_by_14/'

    numpy.save(base_path + 'train_x.npy', train_x)
    numpy.save(base_path + 'train_y.npy', train_y)
    numpy.save(base_path + 'valid_x.npy', valid_x)
    numpy.save(base_path + 'valid_y.npy', valid_y)
    numpy.save(base_path + 'test_x.npy', test_x)
    numpy.save(base_path + 'test_y.npy', test_y)


def load_mnist_multitasking():
    print
    'loading MNIST...'

    print
    'binary MNIST, not the standard split.'
    train = mnist.MNIST('train', binarize=True)
    test = mnist.MNIST('test', binarize=True)
    train_all_x = train.X
    train_all_y = train.y

    def get_one_digit(digit):
        idx = train_all_y == digit
        x = train_all_x[idx]
        y = train_all_y[idx]
        return x, y

    digits = range(10)
    indices = []
    for digit in digits:
        indices.append(train_all_y == digit)

    print
    'shuffling the trainset...'
    idx = range(train_all_y.shape[0])
    numpy.random.RandomState(1234)
    numpy.random.shuffle(idx)
    train_x = train_all_x[idx[:50000]].astype('float32')
    train_y = train_all_y[idx[:50000]].astype('int32')
    valid_x = train_all_x[idx[50000:]].astype('float32')
    valid_y = train_all_y[idx[50000:]].astype('int32')
    test_x = test.X.astype('float32')
    test_y = test.y.astype('int32')


def load_mnist(binary=True, standard_split=False,
               shuffle=False, permute_bits_seed=None, xy=False):
    # dataset_path='/u/yaoli/DeepLearningTutorials/data/mnist.pkl.gz'
    # dataset_path='/mnt/scratch/bengio/yaoli001/data/mnist.pkl.gz'
    # dataset_path = '/scratch/yaoli/datasets/mnist/mnist.pkl.gz'
    print
    'loading MNIST...'

    if binary == True:
        print
        'binary MNIST, not the standard split.'
        train = mnist.MNIST('train', binarize=True, one_hot=True)
        test = mnist.MNIST('test', binarize=True, one_hot=True)
        train_all_x = train.X
        train_all_y = numpy.argmax(train.y, axis=1)
        idx = range(train_all_y.shape[0])
        if shuffle:
            print
            'shuffling the trainset...'
            numpy.random.RandomState(1234)
            numpy.random.shuffle(idx)
        train_x = train_all_x[idx[:50000]].astype('float32')
        train_y = train_all_y[idx[:50000]].astype('int32')
        valid_x = train_all_x[idx[50000:]].astype('float32')
        valid_y = train_all_y[idx[50000:]].astype('int32')
        test_x = test.X.astype('float32')
        test_y = test.y.astype('int32')
    else:
        dataset_path = get_rab_dataset_base_path() + 'mnist.pkl.gz'

        f = open_gz_or_pkl_file(dataset_path)
        trainset, validset, testset = cPickle.load(f)
        f.close()

        train_x = trainset[0].astype('float32')
        train_y = trainset[1].astype('int32')
        valid_x = validset[0].astype('float32')
        valid_y = validset[1].astype('int32')
        test_x = testset[0].astype('float32')
        test_y = testset[1].astype('int32')
    if standard_split:
        print
        'combining train and valid, valid is in fact test'
        train_x = numpy.concatenate((train_x, valid_x), axis=0)
        train_y = numpy.concatenate((train_y, valid_y))
        valid_x = test_x
        valid_y = test_y

    if permute_bits_seed is not None:
        print
        'permutating the pixels...'

        idx = RAB_tools.shuffle_idx(train_x.shape[1], permute_bits_seed)
        train_x = train_x[:, idx]
        valid_x = valid_x[:, idx]
        test_x = test_x[:, idx]
    # compute the norm
    # binary version: 103.92
    # import ipdb; ipdb.set_trace()
    # avg_norm = RAB_tools.compute_norm(train_x)

    if xy:
        # concatenate y(label) at the end of x
        all_y = numpy.concatenate((train_y, valid_y, test_y), axis=0)
        one_hot_y = RAB_tools.scaler_to_one_hot(labels=all_y, dim=10)
        train_y_1hot, valid_y_1hot, test_y_1hot = numpy.split(one_hot_y,
                                                              [train_y.shape[0],
                                                               train_y.shape[0] + valid_y.shape[0]]
                                                              )

        assert numpy.all(numpy.argmax(train_y_1hot, axis=1).flatten() == train_y)
        assert numpy.all(numpy.argmax(valid_y_1hot, axis=1).flatten() == valid_y)
        assert numpy.all(numpy.argmax(test_y_1hot, axis=1).flatten() == test_y)

        train_x = numpy.concatenate((train_x, train_y_1hot), axis=1).astype('float32')
        valid_x = numpy.concatenate((valid_x, valid_y_1hot), axis=1).astype('float32')
        test_x = numpy.concatenate((test_x, test_y_1hot), axis=1).astype('float32')

    print
    'trainset size: ', train_x.shape
    print
    'validset size: ', valid_x.shape
    print
    'testset size: ', test_x.shape

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_MNIST_batches():
    data_path = '/data/lisatmp2/yaoli/datasets/mnist_batches/'
    batch_idx_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_idx_valid = [11, 12]
    batch_idx_test = [13, 14]
    file_prefix = data_path + 'data_batch_'

    return file_prefix, batch_idx_train, batch_idx_valid, batch_idx_test


def open_gz_or_pkl_file(path):
    _, extension = os.path.splitext(path)
    if extension == '.gz':
        f = gzip.open(path, 'rb')
    elif extension == '.pkl':
        f = open(path, 'rb')
    else:
        NotImplementedError('file format not supported!')
    return f


def unit_test():
    # test1
    # Note that we specify the chunk size where minibatches are loaded.
    # If chunk_size % minibatche_size != 0, we pass the tail all at once
    # when reaching the end of the chunk.
    # train,train_label,valid,valid_label,test,test_label = load_mnist()
    train, train_label, valid, valid_label, test, test_label = load_cifar10_raw()
    import ipdb;
    ipdb.set_trace()
    validset_label_provider = DataProvider_FitMemory(
        dataset=valid_label,
        dataset_name='mnist validset labels', minibatch_size=11)

    dataset = [[], [], []]
    for i in range(3):
        # do 3 epochs
        validset_label_provider.refresh()
        while True:
            minibatch = validset_label_provider.get_a_minibatch()
            if minibatch == 'end of minibatches':
                break
            else:
                # get the training labels
                dataset[i].append(minibatch.tolist())
        print
        'one epoch is done'
        dataset[i] = RAB_tools.flatten_list_of_list(dataset[i])

    for i in range(3):
        assert sum(valid_label == numpy.array(dataset[i])) == valid_label.shape[0]


def load_dataset_as_theano_shared(which=None):
    if which == 'MNIST':
        train, train_label, valid, valid_label, test, test_label = load_mnist()
    else:
        NotImplementedError('dataset not supported')

    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype='float32'))
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype='float32'))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset([test, test_label])
    valid_set_x, valid_set_y = shared_dataset([valid, valid_label])
    train_set_x, train_set_y = shared_dataset([train, train_label])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_google_faces(all=False):
    print
    'loading google faces'
    from scripts.mirzamom.singleframe.googleFaceDataset import GoogleDataset
    D = GoogleDataset()
    x = D.X.astype('float32')
    y = numpy.argmax(D.y, axis=1).astype('int32')
    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(x.shape[0])
    train_x = x[idx1]
    train_y = y[idx1]
    valid_x = x[idx2]
    valid_y = y[idx2]
    test_x = x[idx3]
    test_y = x[idx3]
    if all:
        return x, y
    else:
        return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_google_faces_and_emotion_challenge():
    from scripts.chandiar.afew2_facetubes import AFEW2FaceTubes
    g_x, g_y = load_google_faces(all=True)
    sequence_length = 1
    print
    'loading smooth face tubes from Raul'
    train = AFEW2FaceTubes(which_set='train', sequence_length=sequence_length,
                           greyscale=True,
                           preproc=['smooth', 'remove_background_faces'], size=(48, 48))

    valid = AFEW2FaceTubes(which_set='valid', sequence_length=sequence_length,
                           greyscale=True,
                           preproc=['smooth', 'remove_background_faces'], size=(48, 48))
    s_train_x = train.X / 255.0
    s_train_y = numpy.argmax(train.y, axis=1)
    s_valid_x = valid.X / 255.0
    s_valid_y = numpy.argmax(valid.y, axis=1)
    import ipdb;
    ipdb.set_trace()
    train_x = numpy.concatenate((g_x, s_train_x), axis=0).astype('float32')
    train_y = numpy.concatenate((g_y, s_train_y), axis=0).astype('int32')
    train_x = g_x.astype('float32')
    train_y = g_y.astype('float32')
    valid_x = s_valid_x.astype('float32')
    valid_y = s_valid_y.astype('int32')
    test_x = valid_x
    test_y = valid_y

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def format_dataset_for_nnet_gpu(c_ordered=True):
    """
    This function takes the entire dataset and split it into batches each of which
    will fit into the memory.
    The format is kept such that it is compatible with the GPU version of nnet.
    """
    # save_path = '/data/lisa/data/ubi/gro/datasets_predicting_fun/dataset_in_batches/'
    # save_path = '/data/lisatmp2/yaoli/datasets/mnist_batches/'

    # save_path = '/data/lisa/data/ubi/gro/datasets_predicting_fun/dataset_in_larger_batches/'
    # save_path = '/Tmp/yaoli/dataset_in_batches_no_4_star/'

    # save_path = '/tmp/yaoli/dataset_in_batches_no_4_star/'
    # save_path = get_rab_dataset_base_path() + 'gro_fun/fun_1008_batches/batches_in_20000/'
    # save_path = get_rab_dataset_base_path() + 'faces_google_and_challenge_grayscale/'
    save_path = get_rab_dataset_base_path() + 'face_tubes_batches/'
    # train_x, train_y, valid_x, valid_y, test_x, test_y = load_GRO_fun_1008()
    # train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist()
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_face_tubes()
    # train_x, train_y, valid_x, valid_y, test_x, test_y = load_google_faces_and_emotion_challenge()
    import ipdb;
    ipdb.set_trace()
    n_vis = 2304
    # n_vis=28*28
    # 10k is optimal
    k = 10000
    # this option duplicate samples in the last batch to get k
    padding = True
    n_batches_train = train_x.shape[0] / k if train_x.shape[0] % k == 0 else (train_x.shape[0] / k + 1)
    print
    'about to create %d train batches' % n_batches_train
    if valid_x is not None:
        n_batches_valid = valid_x.shape[0] / k if valid_x.shape[0] % k == 0 else (valid_x.shape[0] / k + 1)
    else:
        n_batches_valid = 0
    print
    'about to create %d valid batches' % n_batches_valid
    n_batches_test = test_x.shape[0] / k if test_x.shape[0] % k == 0 else (test_x.shape[0] / k + 1)
    print
    'about to create %d test batches' % n_batches_test
    label_names = ['0', '1', '2', '3', '4', '5', '6']
    # compose the dataset for nnet_gpu
    meta = {}
    meta['data_mean'] = numpy.zeros((n_vis, 1)).astype('float32')
    meta['label_names'] = label_names
    meta['num_cases_per_batch'] = k
    meta['num_vis'] = n_vis
    meta_file = save_path + 'batches.meta'
    RAB_tools.dump_pkl(meta, meta_file)

    batch_file_base = save_path + 'data_batch_'
    for i in range(n_batches_train):
        batch = {}
        extra_x = None
        extra_y = None
        batch['batch_label'] = 'training batch %d of %d' % (i + 1, n_batches_train)
        s = i * k
        if (i + 1) * k <= train_x.shape[0]:
            e = (i + 1) * k
        else:
            e = train_x.shape[0]
            extra_e = k - (e - s)
            idx = range(train_x.shape[0])
            numpy.random.shuffle(idx)
            use_idx = idx[:extra_e]
            extra_x = train_x[use_idx]
            extra_y = train_y[use_idx]
            print('Uneven splitting, last batch has %d samples rather than %d' % (e - s, k))
        if c_ordered:
            batch['data'] = train_x[s:e, :].T
        else:
            batch['data'] = train_x[s:e, :]
        batch['labels'] = train_y[s:e, ].tolist()
        batch['filenames'] = ''

        if padding and extra_x is not None and extra_y is not None:
            print
            'Padding %d samples for the last batch' % len(extra_y)
            if c_ordered:
                extra_x = extra_x.T
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=1)
            else:
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=0)
            batch['labels'] = batch['labels'] + extra_y.tolist()
        print
        'saving training batch %d of %d' % (i + 1, n_batches_train)
        RAB_tools.dump_pkl(batch, batch_file_base + str(i + 1))

    print
    '--------------------------------------------------------------'
    for i in range(n_batches_valid):
        batch = {}
        extra_x = None
        extra_y = None
        batch['batch_label'] = 'valid batch %d of %d' % (i + 1, n_batches_valid)
        s = i * k
        if (i + 1) * k <= valid_x.shape[0]:
            e = (i + 1) * k
        else:
            e = valid_x.shape[0]
            extra_e = k - (e - s)
            idx = range(valid_x.shape[0])
            numpy.random.shuffle(idx)
            use_idx = idx[:extra_e]
            extra_x = valid_x[use_idx]
            extra_y = valid_y[use_idx]
            print('Uneven splitting, last batch has %d samples rather than %d' % (e - s, k))
        if c_ordered:
            batch['data'] = valid_x[s:e, :].T
        else:
            batch['data'] = valid_x[s:e, :]
        batch['labels'] = valid_y[s:e, ].tolist()
        batch['filenames'] = ''

        if padding and extra_x is not None and extra_y is not None:
            print
            'Padding %d samples for the last batch' % len(extra_y)
            if c_ordered:
                extra_x = extra_x.T
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=1)
            else:
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=0)
            batch['labels'] = batch['labels'] + extra_y.tolist()

        print
        'saving valid batch %d of %d' % (i + 1, n_batches_valid)
        RAB_tools.dump_pkl(batch, batch_file_base + str(n_batches_train + i + 1))

    print
    '-------------------------------------------------------------------'
    for i in range(n_batches_test):
        batch = {}
        extra_x = None
        extra_y = None
        batch['batch_label'] = 'test batch %d of %d' % (i + 1, n_batches_test)
        s = i * k
        if (i + 1) * k <= test_x.shape[0]:
            e = (i + 1) * k
        else:
            e = test_x.shape[0]
            extra_e = k - (e - s)
            idx = range(test_x.shape[0])
            numpy.random.shuffle(idx)
            use_idx = idx[:extra_e]
            extra_x = test_x[use_idx]
            extra_y = test_y[use_idx]
            print('Uneven splitting, last batch has %d samples rather than %d' % (e - s, k))
        if c_ordered:
            batch['data'] = test_x[s:e, :].T
        else:
            batch['data'] = test_x[s:e, :]
        batch['labels'] = test_y[s:e, ].tolist()
        batch['filenames'] = ''

        if padding and extra_x is not None and extra_y is not None:
            print
            'Padding %d samples for the last batch' % len(extra_y)
            if c_ordered:
                extra_x = extra_x.T
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=1)
            else:
                batch['data'] = numpy.concatenate((batch['data'], extra_x), axis=0)
            batch['labels'] = batch['labels'] + extra_y.tolist()

        print
        'saving test batch %d of %d' % (i + 1, n_batches_test)
        RAB_tools.dump_pkl(batch, batch_file_base + str(n_batches_train +
                                                        n_batches_valid + i + 1))


def preprocess_gro_fun_1008(directory='lisatmp'):
    if directory == 'lisatmp2':
        base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_1008_labeled_and_unlabeled/'
    else:
        base_path = '/data/lisatmp/yaoli/datasets/gro_fun/ob_partial_1008_labeled_and_unlabeled/'
    train_x_path = base_path + 'train_x_labeled.npy'
    train_y_path = base_path + 'train_y_labeled.npy'
    test_x_path = base_path + 'test_x_labeled.npy'
    test_y_path = base_path + 'test_y_labeled.npy'

    if os.path.exists(train_x_path):
        print
        'trainset already existed.'
        train_x_all = numpy.load(base_path + 'train_x_labeled.npy')
        train_y_all = numpy.load(base_path + 'train_y_labeled.npy')
    else:
        print
        'making train_x and train_y'
        train_x_part1 = numpy.load(base_path + 'trainset_attrs_part1.npy').astype('float32')
        train_x_part2 = numpy.load(base_path + 'trainset_attrs_part2.npy').astype('float32')
        train_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_funs.pkl')).astype('int32')

        train_x_all = numpy.concatenate((train_x_part1, train_x_part2))

        n_minibatch, n_players, attr = train_x_all.shape
        train_x_all = train_x_all.reshape((n_minibatch * n_players, attr))

        train_y_all = train_y_all.flatten()
        effective = train_y_all != -1
        train_x_all = train_x_all[effective]
        train_y_all = train_y_all[effective]
        numpy.save(train_y_path, train_y_all)
        numpy.save(train_x_path, train_x_all)

    if os.path.exists(test_x_path):
        print
        'testset already existed.'
        test_x_all = numpy.load(base_path + 'test_x_labeled.npy')
        test_y_all = numpy.load(base_path + 'test_y_labeled.npy')
    else:
        print
        'making test_x and test_y'
        test_x_all = numpy.load(base_path + 'testset_attrs.npy').astype('float32')
        a, b, c = test_x_all.shape
        test_x_all = test_x_all.reshape((a * b, c))
        test_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_funs.pkl')).astype('int32')
        test_y_all = test_y_all.flatten()
        effective = test_y_all != -1
        test_x_all = test_x_all[effective]
        test_y_all = test_y_all[effective]
        numpy.save(test_y_path, test_y_all)
        numpy.save(test_x_path, test_x_all)

    all_x = numpy.concatenate((train_x_all, test_x_all))
    all_y = numpy.concatenate((train_y_all, test_y_all))
    print
    'total amount of examples: ', all_x.shape[0]
    import pdb;
    pdb.set_trace()
    if 1:
        train_idx, valid_idx, test_idx = RAB_tools.divide_to_3_folds(all_x.shape[0],
                                                                     mode=[.70, .15, .15])
        train_x = all_x[train_idx]
        train_y = all_y[train_idx]
        valid_x = all_x[valid_idx]
        valid_y = all_y[valid_idx]
        test_x = all_x[test_idx]
        test_y = all_y[test_idx]
        print
        'saving...'

        """
        numpy.savez(get_rab_dataset_base_path() + 'gro_fun/fun_1008_labeled_split_70_15_15/all.npz',
                    train_x=train_x, train_y=train_y,
                    valid_x=valid_x, valid_y=valid_y,
                    test_x=test_x, test_y=test_y,
                    train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
        """
        if directory == 'lisatmp2':
            base_path = get_rab_dataset_base_path() + 'gro_fun/'
        else:
            base_path = '/data/lisatmp/yaoli/datasets/gro_fun/'
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/train_x.npy', train_x)
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/train_y.npy', train_y)
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/valid_x.npy', valid_x)
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/valid_y.npy', valid_y)
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/test_x.npy', test_x)
        numpy.save(base_path + 'fun_1008_labeled_split_70_15_15/test_y.npy', test_y)
        numpy.savez(base_path + 'fun_1008_labeled_split_70_15_15/splitting_indices.npz',
                    train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)


def preprocess_gro_fun_pids():
    base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_744_labeled_and_unlabeled/'
    trainset_funs = RAB_tools.list_of_array_to_array(RAB_tools.load_pkl(base_path + 'trainset_funs.pkl'))
    testset_funs = RAB_tools.list_of_array_to_array(RAB_tools.load_pkl(base_path + 'testset_funs.pkl'))
    trainset_pids = RAB_tools.list_of_array_to_array(RAB_tools.load_pkl(base_path + 'trainset_pids.pkl'))
    testset_pids = RAB_tools.list_of_array_to_array(RAB_tools.load_pkl(base_path + 'testset_pids.pkl'))

    all_ids = numpy.concatenate((trainset_pids, testset_pids))
    all_funs = numpy.concatenate((trainset_funs, testset_funs))
    assert all_ids.shape == all_funs.shape
    ids = all_ids[all_funs != -1].astype('int32')
    # ids.max = 170772
    import ipdb;
    ipdb.set_trace()
    train_idx, valid_idx, test_idx = RAB_tools.divide_to_3_folds(ids.shape[0], mode=[.70, .15, .15])
    numpy.save(base_path + 'funs_all_883843.npy', all_funs[all_funs != -1].astype('int32'))
    numpy.save(base_path + 'pids_all_with_funs_883848.npy', ids)
    numpy.save(base_path + 'train_pid.npy', ids[train_idx])
    numpy.save(base_path + 'valid_pid.npy', ids[valid_idx])
    numpy.save(base_path + 'test_pid.npy', ids[test_idx])
    # 883848 pids

    import ipdb;
    ipdb.set_trace()


def preprocess_gro_fun_744_from_all():
    print
    'loading from 744...'
    """
    base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_744_labeled_and_unlabeled/'
    test_x_744 = numpy.load(base_path + 'testset_attrs.npy').astype('float32')
    n_minibatch, n_players, attr = test_x_744.shape
    test_x_744 = test_x_744.reshape((n_minibatch*n_players, attr))
    test_y_744 = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_funs.pkl')).astype('int32')
    test_y_744 = test_y_744.flatten()
    effective = test_y_744 != -1
    test_y_744 = test_y_744[effective]
    test_x_744 = test_x_744[effective]
    print 'loading from 1008...'
    base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_1008_labeled_and_unlabeled/'
    test_x_1008 = numpy.load(base_path + 'testset_attrs.npy').astype('float32')
    n_minibatch, n_players, attr = test_x_1008.shape
    test_x_1008 = test_x_1008.reshape((n_minibatch*n_players, attr))
    test_y_1008 = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_funs.pkl')).astype('int32')
    test_y_1008 = test_y_1008.flatten()
    effective = test_y_1008 != -1
    test_x_1008 = test_x_1008[effective]
    test_y_1008 = test_y_1008[effective]
    """
    test_x_744 = numpy.load('../fun_attrs_744.npy')
    test_x_1008 = numpy.load('../fun_attrs_1008.npy')
    import ipdb;
    ipdb.set_trace()
    zero_attributes = []
    for i in range(test_x_1008.shape[1]):
        a = test_x_1008[:, i]
        if a.sum() == 0:
            zero_attributes.append(i)
    old_attributes = []
    for i in range(test_x_744.shape[1]):
        print
        i
        a = test_x_744[:, i]
        if a.sum() == 0:
            continue
        found = False
        for j in range(test_x_1008.shape[1]):
            b = test_x_1008[:, j]
            sign = numpy.sum(a - b) == 0
            if sign:
                old_attributes.append(j)
                found = True
                # break

        if not found:
            print
            'not found'
    t = set(range(1008)) - set(zero_attributes).union(set(old_attributes))

    import ipdb;
    ipdb.set_trace()


def make_3D_corkscrew():
    # make the bigger sorkscrew
    rng_numpy, _ = RAB_tools.get_two_rngs()
    k = 5000
    t = numpy.linspace(0, 10, num=k) * numpy.pi
    t = t.reshape((t.shape[0], 1))
    x = numpy.cos(t)
    y = numpy.sin(t)
    z = t
    data = numpy.concatenate([x, y, t], axis=1)
    N = 10
    all_data_bigger = []
    for i in range(N):
        noise = rng_numpy.normal(0, 0.01, (k, 3))
        data = data + noise
        all_data_bigger.append(data)
    all_data_bigger = numpy.concatenate(all_data_bigger, axis=0)

    # make the smaller corkscrew
    shift = 0.
    t = numpy.linspace(0, 10, num=k) * numpy.pi
    t = t.reshape((t.shape[0], 1))
    x = 0.5 * numpy.cos(t) - shift
    y = 0.5 * numpy.sin(t)
    z = t
    data = numpy.concatenate([x, y, t], axis=1)

    N = 10
    all_data_smaller = []
    for i in range(N):
        noise = rng_numpy.normal(0, 0.01, (k, 3))
        data = data + noise
        all_data_smaller.append(data)
    all_data_smaller = numpy.concatenate(all_data_smaller, axis=0)
    label_smaller = numpy.zeros((all_data_smaller.shape[0],))
    label_bigger = numpy.zeros((all_data_bigger.shape[0],)) + 1
    data = numpy.concatenate([all_data_bigger, all_data_smaller], axis=0)

    labels = numpy.concatenate([label_smaller, label_bigger], axis=0).astype('int8')

    def visualize(data, labels, save_path=None):
        # visualization
        RAB_tools.plot_in_3d(data, labels)
        for i in range(3):
            if i == 0:
                continue

            pylab.plot(data[:, 0], data[:, i], '.')
            pylab.show()

    data = RAB_tools.zero_mean_unit_variance(data)
    visualize(data, labels)
    data, order = RAB_tools.shuffle_dataset(data)
    labels = labels[order]
    base_path = RAB_tools.get_rab_dataset_base_path()
    save_path = base_path + '3D_corkscrew/'
    numpy.save(save_path + 'data_x.npy', data.astype('float32'))
    numpy.save(save_path + 'data_y.npy', labels.astype('int32'))


def load_2D_straightline():
    print
    'loading 2D straight line'
    # a simple 2D straight line
    N = 20000
    sigma = 0.02
    x = numpy.linspace(-5, 5, N).reshape((N, 1))
    y = (x + 1).reshape((N, 1))
    data = numpy.hstack([x, y])

    data = RAB_tools.zero_mean_unit_variance(data)

    numpy.random.RandomState(1234)
    noise = numpy.random.normal(0, sigma, size=[N, 2])
    data = data + noise
    return data


def load_2D_spiral():
    print
    'loading 2D spiral'
    N = 50000
    sigma = 0.05
    t = numpy.linspace(0, 10 * numpy.pi, N)
    x = (t ** 2.5 * numpy.cos(t)).reshape((N, 1))
    y = (t ** 2.5 * numpy.sin(t)).reshape((N, 1))
    data = numpy.concatenate([x, y], axis=1)

    data = RAB_tools.zero_mean_unit_variance(data)

    numpy.random.RandomState(1234)
    noise = numpy.random.normal(0, sigma, size=[N, 2])
    # data = data + noise
    return data


def load_2D_circle(add_noise=True):
    print
    'loading 2D circle'
    N = 50000
    sigma = 0.05
    t = numpy.linspace(0, 0.5 * numpy.pi, N)
    x = numpy.cos(t).reshape((N, 1))
    y = numpy.sin(t).reshape((N, 1))
    data = numpy.concatenate([x, y], axis=1)

    data = RAB_tools.zero_mean_unit_variance(data)

    numpy.random.RandomState(1234)
    noise = numpy.random.normal(0, sigma, size=[N, 2])

    # data = data + noise
    return data


def load_2D_curves(curve):
    if curve == 'straightline':
        data = load_2D_straightline()
    elif curve == 'spiral':
        data = load_2D_spiral()
    elif curve == 'circle':
        data = load_2D_circle()
    else:
        raise NotImplementedError('%s not supported!' % curve)
    labels = numpy.zeros(data.shape)
    # pylab.plot(data[:,0],data[:,1],'b.')
    # pylab.savefig('2Dspiral.png')

    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(data.shape[0])

    train_x = data[idx1].astype('float32')
    train_y = labels[idx1].astype('int32')
    valid_x = data[idx2].astype('float32')
    valid_y = labels[idx2].astype('int32')
    test_x = data[idx3].astype('float32')
    test_y = labels[idx3].astype('int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_3D_corkscrew():
    print
    "loading 3D corkscrew dataset"
    base_path = RAB_tools.get_rab_dataset_base_path() + '3D_corkscrew/'
    data = numpy.load(base_path + 'data_x.npy')
    labels = numpy.load(base_path + 'data_y.npy')

    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(data.shape[0])
    train_x = data[idx1].astype('float32')
    train_y = labels[idx1].astype('int32')
    valid_x = data[idx2].astype('float32')
    valid_y = labels[idx2].astype('int32')
    test_x = data[idx3].astype('float32')
    test_y = labels[idx3].astype('int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_manifold():
    print
    'load manifold dataset and push range into [0,1]'
    path = '/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/'

    train_x = RAB_tools.load_pkl(path + 'train_samples.pkl')
    valid_x = RAB_tools.load_pkl(path + 'valid_samples.pkl')
    test_x = RAB_tools.load_pkl(path + 'test_samples.pkl')

    x = numpy.concatenate((train_x, valid_x, test_x), axis=0)

    # push values into [0,1]
    x = x + numpy.abs(x).max()
    x = x / (x.max() + .0)
    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(size=x.shape[0])
    train_x = x[idx1].astype('float32')
    valid_x = x[idx2].astype('float32')
    test_x = x[idx3].astype('float32')

    train_y = numpy.zeros((train_x.shape[0],)).astype('int32')
    valid_y = numpy.zeros((valid_x.shape[0],)).astype('int32')
    test_y = numpy.zeros((test_x.shape[0],)).astype('int32')
    # import ipdb; ipdb.set_trace()
    # compute base line cross-entropy: 0.6933
    baseline = numpy.mean(train_x * numpy.log(0.5) + (1 - train_x) * numpy.log(0.5))

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_toy_manifold_dataset(n_dim=10):
    print
    'loading manifold dataset'
    if n_dim == 2:
        path = '/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d2_eig0.1_comp25_001/'
    else:
        # 5K + 5K + 10K
        path = '/data/lisatmp2/alaingui/dae/datasets/gaussian_mixture/d10_eig0.1_comp25_001/'

    train_x = RAB_tools.load_pkl(path + 'train_samples.pkl').astype('float32')
    train_y = numpy.zeros((train_x.shape[0],)).astype('int32')
    valid_x = RAB_tools.load_pkl(path + 'valid_samples.pkl').astype('float32')
    valid_y = numpy.zeros((valid_x.shape[0],)).astype('int32')
    test_x = RAB_tools.load_pkl(path + 'test_samples.pkl').astype('float32')
    test_y = numpy.zeros((test_x.shape[0],)).astype('int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def add_noise_to_discrete_dataset(k, trainset, n_examples, scale):
    # add noise
    random_int = numpy.random.normal(loc=0, scale=scale, size=n_examples).astype('int')
    # random_int = numpy.random.standard_t(df=scale, size=n_examples).astype('int')

    dataset_with_noise = (trainset + random_int) % k
    # import ipdb; ipdb.set_trace()
    # dataset_with_noise = trainset

    return dataset_with_noise


def add_noise_to_continuous_dataset(trainset, n_examples, scale=0.1):
    noise = numpy.random.normal(loc=0, scale=scale, size=n_examples)


def preprocess_gro_fun_744():
    base_path = get_rab_dataset_base_path() + 'gro_fun/ob_partial_744_labeled/'
    train_x = numpy.load(base_path + 'trainset_x.npy')
    test_x = numpy.load(base_path + 'testset_x.npy')

    train_ids = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_pids.pkl'))
    test_ids = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_pids.pkl'))
    train_ids = RAB_tools.list_of_array_to_array(train_ids)
    test_ids = RAB_tools.list_of_array_to_array(test_ids)

    all_ids = numpy.unique(numpy.concatenate((train_ids, test_ids)))
    print
    'total number of different players: ', all_ids.shape[0]
    print
    'number of different players in the trainset: ', numpy.unique(train_ids).shape[0]
    print
    'number of different players in the testset: ', numpy.unique(test_ids).shape[0]

    train_uids = set([a for a in train_ids])
    test_uids = set([a for a in test_ids])
    count = 0
    for id in test_uids:
        if id in train_uids:
            count += 1
    print
    'number of players in the testset but not in the trainset: ', count

    train_x_all = numpy.load(base_path + 'trainset_x.npy').astype('float32')
    train_y_all = cPickle.load(open_gz_or_pkl_file(base_path + 'trainset_y.pkl')).astype('int32')

    print
    'trainset: ', train_x_all.shape
    test_x = numpy.load(base_path + 'testset_x.npy').astype('float32')
    test_y = cPickle.load(open_gz_or_pkl_file(base_path + 'testset_y.pkl')).astype('int32')

    print
    'testset: ', test_x.shape
    print
    'no validation set'

    all_x = numpy.concatenate((train_x_all, test_x))
    all_y = numpy.concatenate((train_y_all, test_y))
    print
    'total amount of examples: ', all_x.shape[0]

    if 1:
        train_idx, valid_idx, test_idx = RAB_tools.divide_to_3_folds(all_x.shape[0],
                                                                     mode=[.70, .15, .15])
        train_x = all_x[train_idx]
        train_y = all_y[train_idx]
        valid_x = all_x[valid_idx]
        valid_y = all_y[valid_idx]
        test_x = all_x[test_idx]
        test_y = all_y[test_idx]
        """
        numpy.savez(base_path + 'fun_744_split_60_15_15/all.npz',
                    train_x=train_x, train_y=train_y,
                    valid_x=valid_x, valid_y=valid_y,
                    test_x=test_x, test_y=test_y,
                    train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
        """
        base_path = get_rab_dataset_base_path() + 'gro_fun/fun_744_labeled_split_70_15_15/'
        numpy.save(base_path + 'train_x.npy', train_x)
        numpy.save(base_path + 'train_y.npy', train_y)
        numpy.save(base_path + 'valid_x.npy', valid_x)
        numpy.save(base_path + 'valid_y.npy', valid_y)
        numpy.save(base_path + 'test_x.npy', test_x)
        numpy.save(base_path + 'test_y.npy', test_y)
        numpy.savez(base_path + 'splitting_indices.npz',
                    train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

        """
        train_size = train_x.shape[0]
        X = numpy.concatenate((train_x, test_x), axis=0)
        X = RAB_tools.normalize(X)
        import ipdb; ipdb.set_trace()
        train_x = X[0:train_size,:]
        test_x = X[train_size:,:]
        numpy.save(base_path + 'train_x_normalized.npy', train_x)
        numpy.save(base_path + 'test_x_normalized.npy', test_x)
        """


def load_mighty_quest():
    """
    (333887, 205)
    (83542, 205)
    (83570, 205)
    """
    print
    'loading mighty quest diff prediction dataset'
    # base_path = '/data/lisa_ubi/hyperquest/data/pack14/cached/small/'
    base_path = get_rab_dataset_base_path() + 'mighty_diff/'
    train_x = numpy.load(base_path + 'train_diff_cached_X.npy').astype('float32')
    train_y = numpy.load(base_path + 'train_diff_cached_Y.npy').astype('int32')
    valid_x = numpy.load(base_path + 'valid_diff_cached_X.npy').astype('float32')
    valid_y = numpy.load(base_path + 'valid_diff_cached_Y.npy').astype('int32')
    test_x = numpy.load(base_path + 'test_diff_cached_X.npy').astype('float32')
    test_y = numpy.load(base_path + 'test_diff_cached_Y.npy').astype('int32')
    """
    from MLDiffTraining.src.dataset import MightyQuest
    train = MightyQuest(which_set='train', task='diff', use_cache=True)
    valid = MightyQuest(which_set='valid', task='diff', use_cache=True)
    test = MightyQuest(which_set='test', task='diff', use_cache=True)
    train_x = numpy.load(base_path + 'train_diff_cached_X.npy').astype('float32')
    train_y = numpy.load(base_path + 'train_diff_cached_Y.npy')[:,0].astype('int32')
    valid_x = numpy.load(base_path + 'valid_diff_cached_X.npy').astype('float32')
    valid_y = numpy.load(base_path + 'valid_diff_cached_Y.npy')[:,0].astype('int32')
    test_x = numpy.load(base_path + 'test_diff_cached_X.npy').astype('float32')
    test_y = numpy.load(base_path + 'test_diff_cached_Y.npy')[:,0].astype('int32')
    train_x = train.X.astype('float32')
    train_y = train.y.astype('int32')
    valid_x = valid.X.astype('float32')
    valid_y = valid.y.astype('int32')
    test_x = test.X.astype('float32')
    test_y = test.y.astype('int32')
    """

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def preprocess_aligned_faces():
    """
    load aligned faces
    """
    base_dir = get_rab_dataset_base_path() + 'aligned_faces/aligned_faces'
    from emotiw.gdesjardins.cf3rbm.data import EmotiwFaces
    train = EmotiwFaces(which_set='Train')
    train.get_frame_pairs_by_clip(n_pairs=10)


def preprocess_facetube_pairs(n_chunks=1):
    print
    'generating training paris from Guillaume facetube prepro pipeline...'
    (get_train_fn, get_valid_fn, get_test_fn,
     filter_method, chunk_size) = load_facetube_TS()
    data_source = TimeSeriesData(get_train_fn, get_valid_fn,
                                 get_test_fn, filter_method, chunk_size)
    train_x = []
    for i in range(n_chunks):
        sys.stdout.write('\rGenerating %5d/%5d examples' % (
            (i + 1) * chunk_size, n_chunks * chunk_size))
        sys.stdout.flush()
        x, y = data_source.get_data_batch(which_set='train')
        train_x.append(x)
    train_x = numpy.concatenate(train_x, axis=0)
    import ipdb;
    ipdb.set_trace()
    print
    'saving npy...'
    save_path = get_rab_dataset_base_path() + 'facetube_pairs_whitened/pairs/'
    numpy.save(save_path + 'train_x.npy', train_x)


def preprocess_face_tube(sequence_length=1, size=(48, 48)):
    print
    'preprocessing face tube: normalizing. and size: ', size
    # from scripts.mirzamom.conv3d.afew2_facetubes import AFEW2FaceTubes

    from scripts.chandiar.afew2_facetubes import AFEW2FaceTubes
    print
    'loading train...'
    train = AFEW2FaceTubes(which_set='train', sequence_length=sequence_length,
                           greyscale=True,
                           preproc=['smooth', 'remove_background_faces'], size=size)
    print
    'loading valid...'
    valid = AFEW2FaceTubes(which_set='valid', sequence_length=sequence_length,
                           greyscale=True,
                           preproc=['smooth', 'remove_background_faces'], size=size)

    import ipdb;
    ipdb.set_trace()

    image_tiler.visualize_grayscale_img(train.X.astype('float32'), how_many=2500, img_shape=[48, 48])
    # image_tiler.visualize_color_img(train.X.astype('float32'), how_many=2500, img_shape=[48,48], channel_length=48*48)

    c = 1
    print
    'divided by ', c

    train_x = (train.X / c).astype('float32')
    train_y = train.y.argmax(axis=1).astype('int32')
    valid_x = (valid.X / c).astype('float32')
    valid_y = valid.y.argmax(axis=1).astype('int32')

    print
    'number of training examples:', train_x.shape
    print
    'number of validation examples:', valid_x.shape

    print
    'saving...'
    save_path = get_rab_dataset_base_path() + 'face_tubes/'
    numpy.save(save_path + 'train_x.npy', train_x)
    numpy.save(save_path + 'train_y.npy', train_y)
    numpy.save(save_path + 'valid_x.npy', valid_x)
    numpy.save(save_path + 'valid_y.npy', valid_y)

    import ipdb;
    ipdb.set_trace()


def load_facetube_pairs():
    base_path = get_rab_dataset_base_path() + 'facetube_pairs_whitened/pairs/train_x.npy'
    D = numpy.load(base_path)
    n_examples, dim = D.shape
    train_feature_x = D[:, :dim / 2]
    train_feat5Bure_y = D[:, dim / 2:]
    # image_tiler.visualize_facetube_pairs(train_feature_x, train_feature_y, how_many=100)
    print
    'loaded %d facetube pairs' % n_examples
    # import ipdb; ipdb.set_trace()
    return train_feature_x, train_feature_y


def load_face_tubes():
    print
    'loading face tubes...'
    load_path = get_rab_dataset_base_path() + 'face_tubes/'
    train_x = numpy.load(load_path + 'train_x.npy')
    train_y = numpy.load(load_path + 'train_y.npy')
    valid_x = numpy.load(load_path + 'valid_x.npy')
    valid_y = numpy.load(load_path + 'valid_y.npy')
    print
    'trainset: ', train_x.shape
    print
    'validset: ', valid_x.shape
    print
    'testset is the validset!!!'

    print
    'shuffling trainset'
    idx = range(train_x.shape[0])
    numpy.random.shuffle(idx)
    train_x = train_x[idx]
    train_y = train_y[idx]

    test_x = valid_x
    test_y = valid_y
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_shifted_images(full=False):
    print
    'loading shifted images dataset'
    # 13*13 shifted images used in Roland's paper.
    base_path = get_rab_dataset_base_path() + 'shifted_images/all_10k_169_plus_169.npy'
    D = numpy.load(base_path)

    # D = RAB_tools.normalize(D)

    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(D.shape[0])

    train_x = D[idx1]
    train_y = numpy.zeros((len(idx1),)).astype('int32')
    valid_x = D[idx2]
    valid_y = numpy.zeros((len(idx2),)).astype('int32')
    test_x = D[idx3]
    test_y = numpy.zeros((len(idx3),)).astype('int32')
    if full:
        full_x = D[:, :169]
        full_y = D[:, 169:]
        return full_x, full_y
    else:
        if 1:
            train_x = D.astype('float32')
            train_y = numpy.zeros((D.shape[0],)).astype('int32')
            valid_x = train_x
            valid_y = train_y
            test_x = train_x
            test_y = train_y

        return train_x, train_y, valid_x, valid_y, test_x, test_y


def preprocess_berkeley():
    # preprocess the berkeley dataset into numpy object arrays
    # e.g. each row is one [image, segs, contours]
    base_path = RAB_tools.get_rab_dataset_base_path()
    data_path = 'berkeley_segmentation/B500/BSDS500/data/groudTruth/'
    ground_truth_path = base_path + 'berkeley_segmentation/B500/BSDS500/data/groundTruth/'
    image_path = base_path + 'berkeley_segmentation/B500/BSDS500/data/images/'
    save_path = base_path + 'berkeley_segmentation/B500/BSDS500/data/numpy/'
    RAB_tools.create_dir_if_not_exist(save_path)

    def process_ground_truth():
        sets = ['train', 'val', 'test']
        dicts = [dict(), dict(), dict()]
        for i, name in enumerate(sets):
            data_dict = dicts[i]
            files = glob.glob(ground_truth_path + "%s/*.mat" % name)
            for j, mat in enumerate(files):
                sys.stdout.write('\rProcessing %5d/%5d examples of %s' % (
                    j + 1, len(files), name))
                sys.stdout.flush()
                id = RAB_tools.get_file_name_from_full_path(mat)
                segs = []
                contours = []
                labels = scipy.io.loadmat(mat)['groundTruth'].T
                for label in labels:
                    label = tuple(label[0][0, 0])
                    seg = label[0]
                    contour = label[1]
                    segs.append(seg)
                    contours.append(contour)
                data_dict[int(id)] = [segs, contours]
            print
        return dicts

    def process_data():
        sets = ['train', 'val', 'test']
        dicts = [dict(), dict(), dict()]
        for i, name in enumerate(sets):
            data_dict = dicts[i]
            files = glob.glob(image_path + "%s/*.jpg" % name)
            for j, this_image_path in enumerate(files):
                sys.stdout.write('\rProcessing %5d/%5d examples of %s' % (
                    j + 1, len(files), name))
                sys.stdout.flush()
                id = RAB_tools.get_file_name_from_full_path(this_image_path)
                image = Image.open(this_image_path)
                try:
                    assert numpy.asarray(image).shape == (321, 481, 3)
                except AssertionError, e:
                    if numpy.asarray(image).shape == (481, 321, 3):
                        pass
                    else:
                        raise AssertionError(e)
                image_gray = image.convert('L')
                data_dict[int(id)] = numpy.asarray(image_gray, dtype='float32')
            print
        return dicts

    def matching_data_and_ground_truth(lists_ground_truth, lists_data):
        print
        'combining data and labels into a object numpy array'
        Ds = []
        for ground_truth, data in zip(lists_ground_truth, lists_data):
            t = []
            for id, truth in ground_truth.iteritems():
                segs = truth[0]
                contours = truth[1]
                pixels = data[id]
                row = numpy.empty(3, dtype=object)
                row[0] = pixels
                row[1] = segs
                row[2] = contours
                t.append(row)

            Ds.append(numpy.asarray(t))
        return Ds

    lists_ground_truth = process_ground_truth()
    lists_data = process_data()
    # each is of the shape (N, 3)
    train, valid, test = matching_data_and_ground_truth(lists_ground_truth, lists_data)
    print
    'saving numpy to %s' % save_path
    numpy.save(save_path + 'train.npy', train)
    numpy.save(save_path + 'valid.npy', valid)
    numpy.save(save_path + 'test.npy', test)


def libsvm_from_numpy(train_x, train_y, save_path=None, skip_zero=False):
    # this function turns numpy ndarray to format supported by libsvm and liblinear
    assert save_path is not None
    assert train_x is not None
    assert train_y is not None
    assert train_x.shape[0] == train_y.shape[0]
    print
    'formatting from numpy to libsvm'

    line = ''
    for x, y, k in zip(train_x, train_y, range(train_x.shape[0])):
        sys.stdout.write('\rProcessing %5d/%5d examples' % (
            k, train_x.shape[0]))
        sys.stdout.flush()

        line += str(int(y))
        for i, v in enumerate(x):
            if skip_zero and v == 0:
                continue
            line += ' ' + str(i + 1) + ':' + str(v)
            if i == (x.shape[0] - 1):
                line += '\n'

    ext_file = open(save_path, 'w')
    ext_file.write(line)
    print
    '\nSuccess! Data is saved %s' % save_path
    return


def load_facetube_TS():
    print
    'loading facetubes from Guillaume...Testset is Validset'
    train_path = '/data/lisatmp/desjagui/data/emotiwfaces/gcn_whitened/train.pkl'
    valid_path = '/data/lisatmp/desjagui/data/emotiwfaces/gcn_whitened/valid.pkl'
    train = RAB_tools.load_pkl(train_path)
    valid = RAB_tools.load_pkl(valid_path)
    import ipdb;
    ipdb.set_trace()
    get_train_fn = train.get_random_framepair_batch
    get_valid_fn = valid.get_random_framepair_batch
    get_test_fn = valid.get_random_framepair_batch
    filter_method = 'A'
    chunk_size = 10000

    if 1:
        n_pairs = 10
        # use lisa_emotiw/emotiw/gdesjardins/cf3rbm/data.py
        train_x, train_y = train.get_frame_pairs_by_clip(n_pairs=n_pairs)
        valid_x, valid_y = valid.get_frame_pairs_by_clip(n_pairs=n_pairs)
        save_path = get_rab_dataset_base_path() + 'aligned_faces_frame_pairs_by_clip_by_emotion/%s_frame_pairs_per_clip/' % str(
            n_pairs)
        RAB_tools.create_dir_if_not_exist(save_path)
        print
        'saving...'
        numpy.save(save_path + 'train_x.npy', train_x)
        numpy.save(save_path + 'valid_x.npy', valid_x)
        numpy.save(save_path + 'train_y_clipIds.npy', train_y)
        numpy.save(save_path + 'valid_y_clipIds.npy', valid_y)

    return get_train_fn, get_valid_fn, get_test_fn, filter_method, chunk_size


# ------------------------------------------------------------------------------
class SingleImageWorker(object):
    '''
    load one image 
    '''

    def __init__(self, which_file, pic_idx, rescale):
        self.which_file = which_file
        self.pic_idx = pic_idx
        self.rescale = rescale
        self.image = Image.open(which_file).resize(rescale, Image.ANTIALIAS)
        self.image_rows, self.image_cols = self.image.size
        self.img_array = numpy.asarray(self.image, dtype='float32')
        # standardization
        self.img_array = (self.img_array - self.img_array.mean()) / self.img_array.std()
        self.rng_numpy, _ = RAB_tools.get_two_rngs()

    def get_patches_random_train_py(self, how_many, patch_size):
        # get random patches from the first half of the image
        patches = []
        x_lim = self.image_rows / 2 - patch_size[0] - 1
        y_lim = self.image_cols - patch_size[1] - 1
        for i in range(how_many):
            x0 = self.rng_numpy.random_integers(low=0, high=x_lim)
            y0 = self.rng_numpy.random_integers(low=0, high=y_lim)
            patch = self.img_array[x0:x0 + patch_size[0], y0:y0 + patch_size[1]]
            patches.append(patch)

        return numpy.asarray(patches).astype('float32')

    def get_test_image(self):
        return self.img_array[(self.image_rows / 2):, ]


class Brodatz(SingleImageWorker):
    def __init__(self, which_file, pic_idx):
        if which_file == 'auto':
            base_path = RAB_tools.get_rab_dataset_base_path() + '/textures/brodatz/'
            which_file = base_path + 'D%s.gif' % pic_idx
        rescale = [320, 320]
        super(Brodatz, self).__init__(which_file, pic_idx, rescale)

    def get_minibatch_train(self, minibatch_size, patch_size):
        minibatch = self.get_patches_random_train_py(
            how_many=minibatch_size, patch_size=patch_size)
        a, b, c = minibatch.shape
        minibatch = minibatch.reshape((a, 1, b, c)).astype('float32')
        return minibatch


def preprocess_texture():
    print
    'preprocess Brodatz texture dataset'
    name = 'texture_6'
    how_many = 10000
    patch_size = (28, 28)
    base_path = RAB_tools.get_rab_dataset_base_path() + '/textures/brodatz/'
    which_one = name.split('_')[-1]
    which_file = base_path + 'D%s.gif' % which_one
    image_engine = SingleImageWorker(which_file)
    save_dir = base_path + 'D%s/' % which_one
    RAB_tools.create_dir_if_not_exist(save_dir)
    patches = image_engine.get_patches_random_train_py(how_many=how_many, patch_size=patch_size)
    save_path = save_dir + '/random_patches_%d_%d_%d.npy' % (
        how_many, patch_size[0], patch_size[1]
    )
    print
    'saving to %s' % save_path
    numpy.save(save_path, patches)

    # some visulization
    t = patches.reshape(how_many, patch_size[0] * patch_size[1])
    image_tiler.visualize_grayscale_img(data=t, img_shape=patch_size)


def load_brodatz_whole(which_one):
    # which_one is a number that indecing which pic
    print
    'loading brodatz pics full resolution'
    base_path = RAB_tools.get_rab_dataset_base_path() + '/textures/brodatz/'
    pic_path = base_path + 'D%s.gif' % which_one
    image = Brodatz(pic_path, which_one)

    return image


def load_brodatz_patches(which_one):
    base_path = RAB_tools.get_rab_dataset_base_path() + '/textures/brodatz/'
    data = numpy.load(base_path + 'D%s/random_patches_10000_28_28.npy' % which_one)
    a, b, c = data.shape
    data = data.reshape(a, b * c)
    labels = numpy.zeros(data.shape)
    idx1, idx2, idx3 = RAB_tools.divide_to_3_folds(data.shape[0])
    train_x = data[idx1].astype('float32')
    train_y = labels[idx1].astype('int32')
    valid_x = data[idx2].astype('float32')
    valid_y = labels[idx2].astype('int32')
    test_x = data[idx3].astype('float32')
    test_y = labels[idx3].astype('int32')

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# ---------------------------------------------------------------------------
def test_dataprovider_online():
    signature = 'facetube_pairs_TS'
    data_engine = DataEngine(signature, 100, 'float32', 'int32', True)
    n_loops = 500  # how many minibatches do we train
    counter = 0
    x, y = data_engine.get_dataset(which='train')

    if x is not None and y is not None:
        print
        'received new x and y to set up shared variable'
    else:
        raise NotImplementedError()
    while counter < n_loops:
        s, e, x, y = data_engine.get_a_minibatch_idx(which='train')
        # print 'minibatch starts at %d, ends at %d'%(s,e)
        if x is not None and y is not None:
            print
            'advanced a chunk, set up shared variable'

        counter += 1

    print
    'end of training'


def test_KFold():
    # passed
    unlabelled_x = (numpy.zeros((10, 2)) - 1).astype('float32')
    unlabelled_y = (numpy.zeros((10, 2)) - 1).astype('int32')
    train_x = numpy.asarray([range(100), range(100)]).T.astype('float32')
    train_y = numpy.asarray([range(100), range(100)]).T.astype('int32')

    train_x = numpy.concatenate([unlabelled_x, train_x], axis=0)
    train_y = numpy.concatenate([unlabelled_y, train_y], axis=0)
    dp = DataProvider_KFold(train_x, train_y, train_x, train_y, 10, True, True, 10, True)
    xs = []
    ys = []
    v_x = []
    v_y = []
    x, y = dp.get_dataset('train')
    a, b = dp.get_dataset('valid')
    xs.append(x)
    ys.append(y)
    v_x.append(a)
    v_y.append(b)
    for i in range(10):
        dp.next_fold()
        x, y = dp.get_dataset('train')
        a, b = dp.get_dataset('valid')
        xs.append(x)
        ys.append(y)
        v_x.append(a)
        v_y.append(b)
    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    # test_dataprovider_online()
    # format_dataset_for_nnet_gpu()
    # preprocess_gro_fun_1008()
    # preprocess_gro_fun_744()
    # preprocess_gro_fun()
    # load_GRO_winner()
    # load_GRO_fun_1008()
    # preprocess_GRO_kill_ratio()
    # load_GRO_fun_1008_labeled()
    # load_GRO_kill_ratio()
    # preprocess_gro_fun_pids()
    # load_GRO_fun_labeled(which='1008', use_differential_targets=True)
    # preprocess_GRO_kill_ratio_1008()
    # save_gro_fun_just_extra_attrs()
    # preprocess_gro_fun_744_from_all()
    # load_GRO_kill_ratio_calibration()
    load_TFD(style='all_unsupervised')
    # load_mnist()
    # get_toy_manifold_dataset()
    # load_mighty_quest()
    # preprocess_face_tube()
    # load_facetube_pairs()
    # preprocess_facetube_pairs()
    # preprocess_aligned_faces()
    # load_facetube_TS()
    # load_google_faces()
    # load_google_faces_and_emotion_challenge()
    # preprocess_mnist_scaled()
    # test_KFold()
    # load_sin()
    # load_biodesix([0])
    # make_3D_corkscrew()
    # load_3D_corkscrew()
    # load_2D_curves('spiral')
    # preprocess_texture()
    # preprocess_berkeley()
    pass
