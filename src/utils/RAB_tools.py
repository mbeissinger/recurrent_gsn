#!/usr/bin/env python

import cPickle
import sys, logging
import tarfile
import os, re, subprocess
from os.path import basename
import theano
import theano.tensor as T
from theano import ProfileMode
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy
import matplotlib

# make a plot without needing an X-server at all
# this is for saving .png
matplotlib.use('Agg')
# this is for interactive
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import TextArea, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
import locale

locale.setlocale(locale.LC_NUMERIC, "")
# import ipdb
import Image
import contextlib
# from data_tools import image_tiler
# from hinton_demo import hinton
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from scipy import ndimage

theano.config.floatX = 'float32'
floatX = theano.config.floatX

relu = lambda x: T.maximum(0.0, x)

monitor = []


# activations
def sigmoid_py_(x):
    return 0.5 * numpy.tanh(0.5 * x) + 0.5


def softmax_py(x):
    return numpy.exp(x) / numpy.exp(x).sum()


def sigmoid_py(x):
    return 1. / (1 + numpy.exp(-x))


def tanh_py(x):
    return (1 - numpy.exp(-2 * x)) / (1 + numpy.exp(-2 * x))


def apply_act_py(x, act):
    if act == 'sigmoid':
        return sigmoid_py(x)
    elif act == 'tanh':
        return tanh_py(x)
    elif act == 'linear':
        return x
    else:
        raise NotImplementedError('%s not supported!' % act)


def noisy_tanh_py(x):
    noise1 = numpy.random.normal(loc=0, scale=2, size=x.shape)
    noise2 = numpy.random.normal(loc=0, scale=2, size=x.shape)
    return tanh_py(x + noise1) + noise2


# ------------------------------------------------------------------------------------
# data preprocessing
# ------------------------------------------------------------------------------------
def resample(x, y):
    idx = []
    for i in range(6):
        idx.append(y[:, 1] == i)
    x0 = x[idx[0]];
    y0 = y[idx[0]]
    x1 = x[idx[1]];
    y1 = y[idx[1]]
    x2 = x[idx[2]];
    y2 = y[idx[2]]
    x3 = x[idx[3]];
    y3 = y[idx[3]]
    x4 = x[idx[4]];
    y4 = y[idx[4]]
    x5 = x[idx[5]];
    y5 = y[idx[5]]

    k = x0.shape[0] / (x.shape[0] - x0.shape[0])
    x1, y1 = duplicate_dataset(x1, y1, k)
    x2, y2 = duplicate_dataset(x2, y2, k)
    x3, y3 = duplicate_dataset(x3, y3, k * 3)
    x4, y4 = duplicate_dataset(x4, y4, k * 5)
    x5, y5 = duplicate_dataset(x5, y5, k * 2)

    new_x = [x0, x1, x2, x3, x4, x5]
    new_y = [y0, y1, y2, y3, y4, y5]
    new_x = numpy.concatenate(new_x, axis=0)
    new_y = numpy.concatenate(new_y, axis=0)
    new_x, idx = shuffle_dataset(new_x)
    new_y = new_y[idx]
    return new_x, new_y


def duplicate_dataset(x, y, k):
    assert k != 0
    # duplicate data k times
    print
    'duplicating dataset'
    assert x.ndim == 2
    assert x.shape[0] == y.shape[0]
    t = []
    m = []
    for i in range(k):
        t.append(numpy.copy(x))
        m.append(numpy.copy(y))
    t = numpy.asarray(t)
    m = numpy.asarray(m)

    a, b, c = t.shape
    t = t.reshape((a * b, c))
    a, b, c = m.shape
    m = m.reshape((a * b, c))
    assert t.shape[0] == m.shape[0]
    return t, m


def shuffle_dataset(data):
    idx = shuffle_idx(data.shape[0])
    return data[idx], idx


def shuffle_idx(n, shuffle_seed=1234):
    print
    'shuffling dataset'
    idx = range(n)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(idx)
    return idx


def shuffle_idx_matrix(n, how_many_orderings, shuffle_seed=1234):
    print
    'shuffling dataset'
    idx = range(n)
    idx = []
    numpy.random.seed(shuffle_seed)
    for i in range(how_many_orderings):
        t = range(n)
        numpy.random.shuffle(t)
        idx.append(t)
    return idx


def divide_to_3_folds(size, mode=[.70, .15, .15]):
    """
    this function shuffle the dataset and return indices to 3 folds
    of train, valid, test

    minibatch_size is not None then, we move tails around to accommadate for this.
    mostly for convnet.
    """
    numpy.random.seed(1234)
    indices = range(size)
    numpy.random.shuffle(indices)
    s1 = int(numpy.floor(size * mode[0]))
    s2 = int(numpy.floor(size * (mode[0] + mode[1])))
    s3 = size
    idx_1 = indices[:s1]
    idx_2 = indices[s1:s2]
    idx_3 = indices[s2:]

    return idx_1, idx_2, idx_3


def scaler_to_one_hot(labels, dim):
    enc = OneHotEncoder()
    t = labels.reshape((labels.shape[0], 1))
    enc.fit(t)
    rval = enc.transform(t).todense()
    return rval


def zero_mean_unit_variance(data):
    # zero mean unit variance
    print
    'standardizing dataset'
    return preprocessing.scale(data)


def min_max_scale(data):
    # [0,1]
    print
    'scale to [0,1]'
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def uniformization(data):
    # map data into its normalized rank
    pass


def logarithmization(data):
    rval = numpy.log(data)
    assert not numpy.isnan(rval.sum())
    return rval


def uniformization(inparray, zer=True):
    print
    'uniformization of dataset'
    "Exact uniformization of the inparray (matrix) data"
    # Create ordered list of elements
    listelem = list(numpy.sort(list(set(inparray.flatten()))))
    dictP = {}
    totct = 0
    outarray = numpy.ones_like(inparray)
    # initialize element count
    for i in listelem:
        dictP.update({i: 0})
    # count
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                dictP[inparray[i, j]] += 1
                totct += 1
        else:
            dictP[inparray[i]] += 1
            totct += 1
    # cumulative
    prev = 0
    for i in listelem:
        dictP[i] += prev
        prev = dictP[i]
    # conversion
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                outarray[i, j] = dictP[inparray[i, j]] / float(totct)
        else:
            outarray[i] = dictP[inparray[i]] / float(totct)
    if zer:
        outarray = outarray - dictP[listelem[0]] / float(totct)
        outarray /= outarray.max()
    return outarray


# -----------------------------------------------------------------------------------
def check_monotonic_inc(x):
    # check if elements in x increases monotonically
    dx = numpy.diff(x)
    return numpy.all(dx >= 0)


def git_record_most_recent_commit(save_path):
    print
    'saving git repo info'
    record = subprocess.check_output(["git", "show", "--summary"])
    save_file = save_path + 'repo_info.txt'
    create_dir_if_not_exist(save_path)
    file = open(save_file, 'w')
    file.write(record)


def print_a_line():
    print
    '--------------------------------------------------'


def get_two_rngs():
    rng_numpy = numpy.random.RandomState(1234)
    rng_theano = MRG_RandomStreams(1234)

    return rng_numpy, rng_theano


def get_zero():
    # in case stuff gets pulled out of GPU
    return numpy.zeros((1,)).astype('float32')


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def find_all_numbers_in_string(string):
    # return ['10', '100'] when used on samples_10k_e100.png'
    return re.findall(r"([0-9.]*[0-9]+)", string)


def generate_geometric_sequence(start, end, times):
    # generate a geometric sequence
    # e.g. [1, 2, 4, 6, 8, 16, ...., end]
    print
    'generating a geometric sequence'
    assert start < end
    rval = [start]
    e = start
    while True:
        e = e * times
        if e <= end:
            rval.append(e)
        else:
            break

    if rval[-1] < end:
        rval.append(end)

    return rval


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)


def print_numpy(x, precision=3):
    with printoptions(precision=precision, suppress=True):
        print(x)


def show_theano_graph_size(fn):
    print('graph size:', len(fn.maker.fgraph.toposort()))


def get_file_name_from_full_path(path):
    # only return file name without type, given path
    # e.g. 'xx/xx/xx/a.png' return a
    return basename(path).split('.')[0]


def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('(\-?[0-9]+)', s)]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)


def get_shape_convnet(image_shape, filter_shape, conv_type='standard'):
    if conv_type == 'standard':
        x = image_shape[2] - filter_shape[2] + 1
        y = image_shape[3] - filter_shape[3] + 1
        m = image_shape[0]
        c = filter_shape[0]
        outputs_shape = (m, c, x, y)
    else:
        raise NotImplementedError()
    return outputs_shape


def extract_epoch_number(list_of_file_names):
    # a list of model_params_e-1.pkl, return just the epoch number
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('(\-?[0-9]+)', s)]

    epoch_number = [alphanum_key(l)[1] for l in list_of_file_names]
    return epoch_number


def plot_manifold_samples(sample, data, save_path):
    assert sample.shape[1] == data.shape[1]
    print
    'using first %d samples to generate the plot' % data.shape[0]
    n_dim = data.shape[1]
    sample = sample[:data.shape[0]]
    fig = plt.figure()
    for i in range(n_dim - 1):
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(2, 9, i + 1)
        ax.plot(sample[:, i], sample[:, i + 1], '*r')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('samples')
        ax.set_axis_off()

        ax = fig.add_subplot(2, 9, 9 + i + 1)
        ax.plot(data[:, i], data[:, i + 1], '*b')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)


def plot_manifold_denoising(x, tilde, recons, save_path):
    assert x.shape == tilde.shape
    n_dim = x.shape[1]
    fig = plt.figure()
    for i in range(n_dim - 1):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(3, 9, i + 1)
        ax.plot(x[:, i], x[:, i + 1], '*r')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('samples')
        ax.set_axis_off()

        # plot tilde
        ax = fig.add_subplot(3, 9, 9 + i + 1)
        ax.plot(tilde[:, i], tilde[:, i + 1], '*b')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('data')
        ax.set_axis_off()

        # plot reconstructed x
        ax = fig.add_subplot(3, 9, 18 + i + 1)
        ax.plot(recons[:, i], recons[:, i + 1], '*k')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)


def plot_3Dcorkscrew_denoising(x, tdx, recon, save_path):
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x, tdx, recon]
    n = len(D)
    for i in range(n):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        d = D[i]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], '.')

    plt.savefig(save_path)


def plot_3Dcorkscrew_samples(sample, original, save_path):
    n_dim = sample.shape[1]
    fig = plt.figure()
    D = [sample, original]
    n = len(D)
    for i in range(n):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        d = D[i]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], '.')

    plt.savefig(save_path)


def plot_3Dcorkscrew_denoising(x, tdx, recon, save_path):
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x, tdx, recon]
    n = len(D)
    for i in range(n):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        d = D[i]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], '.')

    plt.savefig(save_path)


def plot_2D_denoising(x, tdx, recon, save_path):
    assert x.shape[1] == 2
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x, tdx, recon]
    n = len(D)

    distances = [0, numpy.sqrt(((D[1] - D[0]) ** 2).sum(axis=1)).mean(),
                 numpy.sqrt(((D[2] - D[0]) ** 2).sum(axis=1)).mean()]
    # pick up some points to tag
    idx = shuffle_idx(x.shape[0])[:5]
    points = []
    for data in D:
        points.append(data[idx, :])

    axs = []
    title = ['trainset', 'corrupted', 'reconstruct']
    for i in range(n):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1, n, i + 1)
        d = D[i]
        ax.scatter(d[:, 0], d[:, 1])
        ax.set_title('%s,%.3f ' % (title[i], distances[i]))
        axs.append(ax)

        # now mark some points
        for i, point in enumerate(points[i]):
            offsetbox = TextArea("%d" % i, minimumdescent=False)
            ab = AnnotationBbox(offsetbox, point,
                                xybox=(-20, 40),
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle='arc3,rad=0.5',
                                                color='r'))
            ax.add_artist(ab)

    x_lim = axs[1].get_xlim()
    y_lim = axs[1].get_ylim()
    for ax in axs:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plt.savefig(save_path)


def plot_2D_samples(sample, original, save_path):
    assert sample.shape[1] == 2
    n_dim = sample.shape[1]
    fig = plt.figure()
    D = [sample, original]
    n = len(D)
    axs = []
    for i in range(n):
        # plot x
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1, n, i + 1)
        d = D[i]
        ax.scatter(d[:, 0], d[:, 1])
        if i == 0:
            ax.set_title('samples')
        if i == 1:
            ax.set_title('trainset')
        axs.append(ax)

    x_lim = axs[0].get_xlim()
    y_lim = axs[0].get_ylim()
    for ax in axs:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plt.savefig(save_path)


def plot_3D_samples(sample, data, save_path):
    assert sample.shape[1] == 10
    assert sample.shape[1] == data.shape[1]
    print
    'using first %d samples to generate the plot' % data.shape[0]
    n_dim = data.shape[1]
    sample = sample[:data.shape[0]]
    fig = plt.figure()
    for i in range(n_dim - 1):
        # fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(2, 9, i + 1)
        ax.plot(sample[:, i], sample[:, i + 1], '*r')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('samples')
        ax.set_axis_off()

        ax = fig.add_subplot(2, 9, 9 + i + 1)
        ax.plot(data[:, i], data[:, i + 1], '*b')
        # ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        # ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)


def apply_act(x, act=None):
    # apply act(x)
    # linear:0, sigmoid:1, tanh:2, relu:3, softmax:4, ultra_fast_sigmoid:5
    if act == 'sigmoid' or act == 1:
        rval = T.nnet.sigmoid(x)
    elif act == 'tanh' or act == 2:
        rval = T.tanh(x)
    elif act == 'relu' or act == 3:
        rval = relu(x)
    elif act == 'linear' or act == 0:
        rval = x
    elif act == 'softmax' or act == 4:
        rval = T.nnet.softmax(x)
    elif act == 'ultra_fast_sigmoid' or act == 5:
        # does not seem to work with the current Theano, gradient not defined!
        rval = T.nnet.ultra_fast_sigmoid(x)
    else:
        raise NotImplementedError()
    return rval


def build_weights(n_row=None, n_col=None, style=None, name=None,
                  rng_numpy=None, value=None, how_many=None):
    # build shared theano var for weights
    if how_many is None:
        size = (n_row, n_col)
    else:
        size = (n_rows, how_many, n_col)

    if value is not None:
        print
        'use existing value to init weights'
        if len(size) == 3:
            assert value.shape == (n_row, how_many, n_col)
        else:
            assert value.shape == (n_row, n_col)
        rval = theano.shared(value=value, name=name)
    else:
        if style == 0:
            # do this only when sigmoid act
            print
            'init %s with FORMULA' % name
            value = numpy.asarray(rng_numpy.uniform(
                low=-4 * numpy.sqrt(6. / (n_row + n_col)),
                high=4 * numpy.sqrt(6. / (n_row + n_col)),
                size=size), dtype=floatX)
        elif style == 1:
            print
            'init %s with Gaussian (0, %f)' % (name, 0.01)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=0.01,
                                                   size=size), dtype=floatX)
        elif style == 2:
            print
            'init with another FORMULA'
            value = numpy.asarray(rng_numpy.uniform(
                low=-numpy.sqrt(6. / (n_row + n_col)),
                high=numpy.sqrt(6. / (n_row + n_col)),
                size=size), dtype=floatX)
        elif style == 3:
            print
            'int weights to be all ones, only for test'
            value = numpy.ones(size, dtype=floatX)
        elif style == 4:
            print
            'usual uniform initialization of weights -1/sqrt(n_in)'
            value = numpy.asarray(rng_numpy.uniform(
                low=-1 / numpy.sqrt(n_row),
                high=1 / numpy.sqrt(n_row), size=size), dtype=floatX)
        else:
            raise NotImplementedError()

        rval = theano.shared(value=value, name=name)
    return rval


def build_bias(size=None, name=None, value=None):
    # build theano shared var for bias
    if value is not None:
        assert value.shape == (size,)
        print
        'use existing value to init bias'
        rval = theano.shared(value=value, name=name)
    else:
        rval = theano.shared(value=numpy.zeros(size, dtype=floatX), name=name)
    return rval


def corrupt_with_masking(x, size, corruption_level, rng_theano):
    rval = rng_theano.binomial(size=size, n=1,
                               p=1 - self.corruption_level,
                               dtype=theano.config.floatX) * x
    return rval


def corrupt_with_salt_and_pepper(x, size, corruption_level, rng_theano):
    a = rng_theano.binomial(size=size, n=1,
                            p=1 - corruption_level,
                            dtype=theano.config.floatX)
    b = rng_theano.binomial(size=size, n=1,
                            p=0.5,
                            dtype=theano.config.floatX)
    c = T.eq(a, 0) * b

    rval = x * a + c
    return rval


def corrupt_with_gaussian(x, size, corruption_level, rng_theano):
    noise = rng_theano.normal(size=size, avg=0.0,
                              std=corruption_level)
    rval = x + noise
    return rval


def cross_entropy_cost(outputs, targets):
    L = - T.mean(targets * T.log(outputs) +
                 (1 - targets) * T.log(1 - outputs), axis=1)
    cost = T.mean(L)
    return cost


def isotropic_gaussian_LL(means_estimated, stds_estimated, targets):
    # the loglikelihood of isotropic Gaussian with
    # estimated mean and std
    A = -((targets - means_estimated) ** 2) / (2 * (stds_estimated ** 2))
    B = -T.log(stds_estimated * T.sqrt(2 * numpy.pi))
    LL = (A + B).sum(axis=1).mean()
    return LL


def isotropic_gaussian_LL_py(means_estimated, stds_estimated, targets):
    raise NotImplementedError()


def mse_cost(outputs, targets, mean_over_second=True):
    if mean_over_second:
        cost = T.mean(T.sqr(targets - outputs))
    else:
        cost = T.mean(T.sqr(targets - outputs).sum(axis=1))
    return cost


def compute_norm(data):
    # data is a matrix (samples, features)
    # compute the avg norm of data
    return (data ** 2).sum(axis=1).mean()


def gaussian_filter(data, sigma):
    return ndimage.gaussian_filter(data, sigma)


def mcmc_autocorrelation(samples):
    assert samples.ndim == 2
    # compute the autocorrelation of samples from MCMC, reference Heng's paper.
    N = samples.shape[0]
    taos = numpy.arange(N / 2)
    vals = []
    for tao in taos:
        idx_a = numpy.arange(N / 2)
        a = samples[idx_a, :]
        b = samples[idx_a + tao, :]
        assert a.shape == b.shape
        numer = (a * b).sum()
        denom_1 = numpy.sqrt((a * a).sum())
        denom_2 = numpy.sqrt((b * b).sum())
        autocorr = (numer + 0.) / (denom_1 * denom_2)
        vals.append(autocorr)
    return vals, taos


def color_to_gray(D):
    img = Image.fromarray(D)
    img_gray = img.convert('L')
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


def get_history(records, k):
    # records: list
    # k: int
    # this function returns the latest k elements of records
    # if records does not have k elements, return what is there
    def rounding(element):
        try:
            if type(element) is list:
                rval = [round(e, 4) for e in element]
            else:
                rval = round(element, 4)
        except TypeError, e:
            print
            e
            import ipdb;
            ipdb.set_trace()
        return rval

    if len(records) < k:
        rval = records[::-1]
    else:
        rval = records[-k:][::-1]

    rval = numpy.asarray(rval).tolist()
    rval = map(rounding, rval)
    return rval


def image2array(im):
    if im.mode not in ("L", "F"):
        raise ValueError, "can only convert single-layer images"
    if im.mode == "L":
        a = Numeric.fromstring(im.tostring(), Numeric.UnsignedInt8)
    else:
        a = Numeric.fromstring(im.tostring(), Numeric.Float32)
    a.shape = im.size[1], im.size[0]
    return a


def array2image(a):
    if a.typecode() == Numeric.UnsignedInt8:
        mode = "L"
    elif a.typecode() == Numeric.Float32:
        mode = "F"
    else:
        raise ValueError, "unsupported image mode"
    return Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())


def resize_img(imgs):
    # imgs is e.g. (60k, 28, 28)
    new_shape = [14, 14]
    new_imgs = []
    for k, img in enumerate(imgs):
        sys.stdout.write('\rResizing %d/%d examples' % (
            k, imgs.shape[0]))
        sys.stdout.flush()
        # if k == 100:
        #    break

        new_img = Image.fromarray(img)
        new_img = new_img.resize(new_shape, Image.ANTIALIAS)
        # new_img.save('/tmp/yaoli/test/mnist_rescaled.png')
        new_imgs.append(numpy.asarray(list(new_img.getdata())))

    mnist_scaled = numpy.asarray(new_imgs).astype('float32')
    # image_tiler.visualize_mnist(data=mnist_scaled, how_many=100, image_shape=new_shape)
    return mnist_scaled


def get_rab_exp_path():
    # LISA: /data/lisa/exp/yaoli/
    return os.environ.get('RAB_EXP_PATH')


def get_rab_dataset_base_path():
    # LISA: /data/lisatmp2/yaoli/datasets/
    return os.environ.get('RAB_DATA_PATH')


def get_rab_model_base_path():
    # LISA: /data/lisatmp2/yaoli/models/
    return os.environ.get('RAB_MODEL_PATH')


def get_parent_dir(path):
    # from '/a/b/c' to '/a/b'
    return '/'.join(path.split('/')[:-1])


def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print
        'creating directory %s' % directory
        os.makedirs(directory)
    else:
        print
        "%s already exists!" % directory


def get_dummy_theano_var_with_test_value(shape, vtype='fmatrix', name='fm'):
    if vtype == 'fmatrix':
        var = T.fmatrix(name)
        var.tag.test_value = numpy.random.binomial(1, 0.5, shape).astype(floatX)

    return var


def get_theano_constant(constant, dtype, bc_pattern):
    # usage: dtype = 'float32', bc_pattern=()
    # see http://deeplearning.net/software/theano/library/tensor/basic.html for details.
    try:
        rval = theano.tensor.TensorConstant(theano.tensor.TensorType(dtype,
                                                                     broadcastable=bc_pattern),
                                            numpy.asarray(constant, 'float32'))
    except TypeError, e:
        print
        e
        import ipdb;
        ipdb.set_trace()
    return rval


def run_with_try(func):
    """
    Call `func()` with fallback in pdb on error.
    """
    try:
        return func()
    except Exception, e:
        print
        '%s: %s' % (e.__class__, e)
        ipdb.post_mortem(sys.exc_info()[2])
        raise


def list_of_array_to_array(l):
    base = l[0]
    sizes = [member.shape[0] for member in l]
    n_samples = sum(sizes)
    # print n_samples
    if l[0].ndim == 2:
        n_attrs = l[0].shape[1]
        X = numpy.zeros((n_samples, n_attrs))
    elif l[0].ndim == 1:
        X = numpy.zeros((n_samples,))
    else:
        NotImplementedError('ndim of the element of the list must be <=2')
    idx_start = 0
    for i, member in enumerate(l):
        # sys.stdout.write('\r%d/%d'%(i, len(l)))
        # sys.stdout.flush()
        if X.ndim == 2:
            X[idx_start:(idx_start + member.shape[0]), :] = member
        else:
            X[idx_start:(idx_start + member.shape[0])] = member
        idx_start += member.shape[0]
    return X


# --------------------------------------------------------------------------------
# the following are universial tools for check MLP training
# It assumes that train_stats.npy is found, only application when models
# are trained by K fold cv

def model_selection_one_exp(model_path):
    # model_path = '/data/lisa/exp/yaoli/test/train_stats.npy'
    stats = numpy.load(model_path)

    train_cost = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')
    train_error = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')
    valid_cost = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')
    valid_error = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')
    test_cost = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')
    test_error = numpy.empty([stats.shape[0], stats.shape[1] - 1], dtype='float64')

    for i in range(stats.shape[0]):
        for j in range(stats.shape[1] - 1):
            value = stats[i, j]
            assert value is not None

            train_cost[i, j] = value['train_cost']
            train_error[i, j] = value['train_error']
            valid_cost[i, j] = value['valid_cost']
            valid_error[i, j] = value['valid_error']
            test_cost[i, j] = value['test_cost']
            test_error[i, j] = value['test_error']

    # epoch selection
    avg_valid_cost = valid_cost.mean(axis=1)
    best_epoch = numpy.argmin(avg_valid_cost)

    retrain_train_cost = stats[best_epoch, -1]['train_cost']
    retrain_train_error = stats[best_epoch, -1]['train_error']
    retrain_test_error = stats[best_epoch, -1]['test_error']
    retrain_test_cost = stats[best_epoch, -1]['test_cost']

    print
    'best epoch %d/%d' % (best_epoch, stats.shape[0])
    print
    'retrain train error ', retrain_train_error
    print
    'retrain test error ', retrain_test_error


def unique_rows(data):
    uniq = numpy.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def print_histogram(data, bins=20):
    counts, intervals = numpy.histogram(data.flatten(), bins=20)
    print_numpy(counts)
    print_numpy(intervals)


def interval_mean(data, n_split):
    # data is numpy array
    interval = numpy.split(numpy.sort(data.flatten())[::-1], n_split)
    means = []
    for i in range(n_split):
        means.append(numpy.mean(interval[:(i + 1)]))
    return means


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def show_random_sampling_graph(data=None, cost=None, name=None):
    # data: matrix, each row is one exp, each col is a hyper-param
    # cost: vector, corresponds to each row(exp) in data
    if 0:
        data = numpy.random.normal(loc=0, scale=1, size=(100, 10))
        cost = numpy.random.normal(loc=0, scale=1, size=(100,))
        name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    assert data.shape[1] == len(name)
    assert data.shape[0] == cost.shape[0]
    shape = int(numpy.ceil(numpy.sqrt(data.shape[1])))
    fig = plt.figure(figsize=[20, 10])
    for i in range(data.shape[1]):
        x = data[:, i]
        y = cost
        ax = plt.subplot(shape, shape, i + 1)
        ax.scatter(x, y)
        ax.set_xscale('log')
        ax.set_xlabel(name[i])

    plt.show()


def plot_scatter_tsne(D, labels):
    fig = plt.figure()
    color = ['r', 'b', 'g', 'k', 'm', 'y']
    axs = []
    names = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
    for i in numpy.unique(labels):
        to_scatter = D[labels == i]
        axs.append(plt.scatter(to_scatter[:, 0], to_scatter[:, 1], s=30, c=color[i]))
    plt.legend(axs, ['0', '1'])
    plt.show()


def plot_one_line(x, xlabel, ylabel, title):
    plt.plot(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_two_lines(x, y, title_x, title_y):
    plt.plot(x, label=title_x)
    plt.plot(y, label=title_y)
    plt.legend()
    plt.show()


def plot_three_lines(x, y, z, legend_x, legend_y, legend_z, title, xlabel, ylabel):
    plt.plot(x, label=legend_x)
    plt.plot(y, label=legend_y)
    plt.plot(z, label=legend_z)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_four_lines(x, y, z, t, legend_x, legend_y, legend_z, legend_t,
                    title, xlabel, ylabel):
    plt.plot(x, label=legend_x)
    plt.plot(y, label=legend_y)
    plt.plot(z, label=legend_z)
    plt.plot(t, label=legend_t)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_scatter(x, y, xlabel, ylabel):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_in_3d(x, labels=None, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    else:
        color = ['r', 'b']
        for i in numpy.unique(labels):
            idx = labels == i
            ax.scatter(x[idx, 0], x[idx, 1],
                       x[idx, 2], '.', color=color[i])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    return ax


def plot_acts_3D(acts):
    assert len(acts) == 3
    fig = plt.figure()
    act = acts[0]
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(act[:, 0], act[:, 1], act[:, 2])

    act = acts[1]
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(act[:, 0], act[:, 1], act[:, 2])

    act = acts[2]
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(act[:, 0], act[:, 1], act[:, 2])
    plt.show()


def histogram_acts(acts, save_path=None, label=None):
    assert len(acts) == len(label)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(acts, bins=20, label=label)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def errorbar_acts(acts, save_path=None, title=None):
    shape = int(numpy.ceil(numpy.sqrt(len(acts))))
    fig = plt.figure(figsize=(12, 10))
    for k, act in enumerate(acts):
        # act in one layer
        ax = fig.add_subplot(len(acts), 1, k + 1)
        x = range(act.shape[1])
        y = numpy.mean(act, axis=0)
        error = numpy.std(act, axis=0)
        ax.errorbar(x, y, error)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def diagram_acts(acts, preacts=None, label=None, save_path=None):
    # generate error bars of activation and the histogram
    # acts is a list of acts
    fig = plt.figure(figsize=(25, 15))

    for k, act in enumerate(acts):
        # act in one layer
        ax = plt.subplot2grid((len(acts) + 3, 4), (k, 0))
        x = range(act.shape[1])
        y = numpy.mean(act, axis=0)
        error = numpy.std(act, axis=0)
        ax.errorbar(x, y, error, color='b', ecolor='r')

        ax = plt.subplot2grid((len(acts) + 3, 4), (k, 1))
        ax.hist(act, bins=10, label=label[k])
        ax.legend()

        cov = numpy.corrcoef(act)
        ax = plt.subplot2grid((len(acts) + 3, 4), (k, 2))
        ax.imshow(cov, cmap=plt.cm.gray)

        ax = plt.subplot2grid((len(acts) + 3, 4), (k, 3))
        x = range(act.shape[1])
        y = numpy.mean(preacts[k], axis=0)
        error = numpy.std(preacts[k], axis=0)
        ax.errorbar(x, y, error, color='b', ecolor='r')

        # ax = plt.subplot2grid((len(acts)+3,5), (k,4))
        # x = range(act.shape[1])
        # y = numpy.mean(preacts[k] - act, axis=0)
        # error = numpy.std(preacts[k] - act,axis=0)
        # ax.errorbar(x, y, error, color='b', ecolor='r')

    ax = plt.subplot2grid((len(acts) + 3, 4), (len(acts), 0), rowspan=3, colspan=4)
    ax.hist(acts, bins=20, label=label)
    ax.legend()

    # fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def diagram_act_single_layer(act, save_path=None):
    act = act[:1000]
    # the following line complains about $DISPLAY
    fig = plt.figure(figsize=(16, 6))
    ax = plt.subplot2grid((1, 4), (0, 0))
    # x = range(act.shape[1])
    # y = numpy.mean(act, axis=0)
    # error = numpy.std(act, axis=0)
    # ax.errorbar(x, y, error, color='b', ecolor='r')
    ax.boxplot(act)
    ax.set_xticks([-1] + range(act.shape[1]) + [act.shape[1]])

    ax = plt.subplot2grid((1, 4), (0, 1))
    ax.hist(act.flatten(), bins=10)
    ax.locator_params(tight=True, nbins=5)

    cov = numpy.corrcoef(act)
    ax = plt.subplot2grid((1, 4), (0, 2))
    ax.matshow(cov, cmap=plt.cm.gray)
    ax.locator_params(tight=True, nbins=5)
    # plt.colorbar()

    ax = plt.subplot2grid((1, 4), (0, 3))
    ax.matshow(act[:50], cmap=plt.cm.gray)
    # plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def histogram_params(params, save_path=None):
    # data is a list of vector or matrices
    shape = int(numpy.ceil(numpy.sqrt(len(params))))
    fig = plt.figure()
    for i, param in enumerate(params):
        value = param.get_value()
        ax = fig.add_subplot(1, 3, i + 1)
        ax.hist(value.flatten(), bins=20)

        ax.set_title('param shape ' + str(value.shape))
        ax.locator_params(tight=True, nbins=5)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def hintondiagram_params(params, save_path=None, shape=None):
    if shape is None:
        n_row = int(numpy.ceil(numpy.sqrt(len(params))))
        n_col = n_row
    else:
        n_row = shape[0]
        n_col = shape[1]

    fig = plt.figure(figsize=(16, 12))
    for i, param in enumerate(params):
        value = param.get_value()
        if value.ndim == 1:
            value = value.reshape((value.shape[0], 1))
        ax = fig.add_subplot(n_row, n_col, i + 1)
        hinton(value, ax=ax)
        ax.set_title('param shape ' + str(value.shape))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def heatmap_params(params, save_path=None, shape=None):
    if shape is None:
        n_row = int(numpy.ceil(numpy.sqrt(len(params))))
        n_col = n_row
    else:
        n_row = shape[0]
        n_col = shape[1]

    fig = plt.figure(figsize=(20, 15))
    for i, param in enumerate(params):
        value = param.get_value()
        if value.ndim == 1:
            value = value.reshape((value.shape[0], 1))
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(value, cmap=plt.cm.gray)
        ax.set_title('param shape ' + str(value.shape))

        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    # fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def diagram_shift(ax):
    # neat trick to move origin right-up
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax


def plot_learning_curves(plt, set_1, set_2,
                         label_1='train NLL', label_2='test NLL',
                         style_1='k-', style_2='k--'):
    plt.plot(set_1, style_1, label=label_1)
    plt.plot(set_2, style_2, label=label_2)
    # plt.legend()
    # plt.show()
    return plt


def plot_learning_curves_from_npz():
    # the file should have 'train_cost', 'valid_cost'
    file_path = sys.argv[1]
    t = numpy.load(file_path)
    plt.plot(t['train_cost'], label='train cost')
    plt.plot(t['valid_cost'], label='valid cost')
    plt.plot(t['test_cost'], label='test cost')
    plt.plot(t['test_error'], label='test error')
    plt.legend()
    plt.show()


def plot_two_vector(x, y):
    plt.plot(x, 'r-')
    plt.plot(y, 'b-')
    plt.show()


def plot_histogram(data, bins, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data.flatten(), bins=bins)
    plt.savefig(save_path)


def plot_two_vector():
    a = [10, 20, 30, 40, 50, 60, 90, 120, 150, 180]
    b = [122, 137, 162, 156, 204, 183, 194, 154, 173, 186]
    c = [128, 164, 133, 99, 129, 124, 131, 144, 109, 132]
    plt.plot(a, b, 'k-*')
    plt.plot(a, c, 'r-*')
    plt.xlabel('epoch')
    plt.ylabel('log-likelihood')
    plt.legend(('9 steps of walkback', 'no walkback'))
    plt.show()


def plot_noisy_tanh():
    x = numpy.linspace(-10, 10, 100)
    x_tanh = tanh_py(x)

    x_noisy_tanhs = []
    for i in range(1000):
        x_noisy_tanh = noisy_tanh_py(x)
        x_noisy_tanhs.append(x_noisy_tanh)

    x_noisy_tanhs = numpy.asarray(x_noisy_tanhs)
    plt.plot(range(len(x)), x_tanh, 'r-')
    plt.plot(range(len(x)), x_noisy_tanhs.sum(axis=0), 'b-')
    plt.show()


def plot_cost_from_npz():
    path_1 = '/data/lisa/exp/yaoli/gsn-2-w5-n04-e100-noise2_h/stats.npz'
    path_2 = '/data/lisa/exp/yaoli/gsn-1-w1-n04-e100/stats.npz'

    t1 = numpy.load(path_1)
    t2 = numpy.load(path_2)
    to_plot_1 = t1['train_cost']
    to_plot_2 = t2['train_cost']
    x = range(len(to_plot_1))
    plt.plot(x, to_plot_1, 'k-')
    plt.plot(x, to_plot_2[:len(x)], 'r-')
    plt.legend(('2 walkback', '1 walkback'))
    plt.show()


def show_grayscale_img(img):
    # img is a matrix with values 0-255
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


# ---------------------------------------------------------------------------------------
def log_sum_exp_theano(x, axis):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), numpy.std(a)
    h = 1.96 * se / numpy.sqrt(n)
    return m, m - h, m + h


def get_profile_mode():
    profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
    return profmode


def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def load_tar_bz2(path):
    """
    Load a file saved with `dump_tar_bz2`.
    """
    assert path.endswith('.tar.bz2')
    name = os.path.split(path)[-1].replace(".tar.bz2", ".pkl")
    f = tarfile.open(path).extractfile(name)
    try:
        data = f.read()
    finally:
        f.close()
    return cPickle.loads(data)


def dump_tar_bz2(obj, path):
    """
    Save object to a .tar.bz2 file.

    The file stored within the .tar.bz2 has the same basename as 'path', but
    ends with '.pkl' instead of '.tar.bz2'.

    :param obj: Object to be saved.

    :param path: Path to the file (must end with '.tar.bz2').
    """
    assert path.endswith('.tar.bz2')
    pkl_name = os.path.basename(path)[0:-8] + '.pkl'
    # We use StringIO to avoid having to write to disk a temporary
    # pickle file.
    obj_io = None
    f_out = tarfile.open(path, mode='w:bz2')
    try:
        obj_str = cPickle.dumps(obj)
        obj_io = StringIO.StringIO(obj_str)
        tarinfo = tarfile.TarInfo(name=pkl_name)
        tarinfo.size = len(obj_str)
        f_out.addfile(tarinfo=tarinfo, fileobj=obj_io)
    finally:
        f_out.close()
        if obj_io is not None:
            obj_io.close()


def pkl_to_hdf5(train_x, valid_x, test_x, hdf5_path):
    print
    'creating %s' % hdf5_path
    import h5py
    data = [train_x, valid_x, test_x]
    name = ['train', 'valid', 'test']
    f = h5py.File(hdf5_path, 'w')
    for x, name in zip(data, name):
        group = f.create_group(name)
        dset = group.create_dataset('data', x.shape, 'f')
        dset[...] = x
    f.close()


def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def flatten(list_of_lists):
    """
    Flatten a list of lists.

    :param list_of_lists: A list of lists l1, l2, l3, ... to be flattened.

    :return: A list equal to l1 + l2 + l3 + ...
    """
    rval = []
    for li in list_of_lists:
        rval += li
    return rval


def center_pixels(X):
    return (X - 127.5) / 127.5


def normalize(X):
    print
    'normalizing the dataset...'

    stds = numpy.std(X, axis=0)
    eliminate_columns = numpy.where(stds == 0)[0]
    if eliminate_columns.size != 0:
        print
        "remove constant columns ", eliminate_columns
        valid_columns = stds != 0
        Y = X[:, valid_columns]
    else:
        Y = X
    stds = numpy.std(Y, axis=0)
    means = numpy.mean(Y, axis=0)

    return (Y - means) / stds


def pca(X, k=2):
    print
    'performing PCA with k=%d ' % k
    pca = PCA(n_components=k)
    pca.fit(X)
    # import ipdb; ipdb.set_trace()
    # print(pca.explained_variance_ratio_)
    return pca


def show_linear_correlation_coefficients(x, y):
    values = []
    if x.ndim == 1:
        print
        numpy.corrcoef(x, y)[0, 1]
    elif x.ndim == 2:
        for idx in range(x.shape[1]):
            values.append(numpy.corrcoef(x[:, idx], y)[0, 1])
    else:
        NotImplementedError
    return values


# the following is the code to print out a pretty table
def format_num(num):
    """Format a number according to given places.
    Adds commas, etc.

    Will truncate floats into ints!"""

    try:
        return '%.2e' % (num)

    except (ValueError, TypeError):
        return str(num)


def get_max_width(table, index):
    """Get the maximum width of the given column index
    """

    return max([len(format_num(row[index])) for row in table])


def pprint_table(table):
    """Prints out a table of data, padded for alignment

    @param out: Output stream ("file-like object")
    @param table: The table to print. A list of lists. Each row must have the same
    number of columns.

    """
    out = sys.stdout

    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print >> out, row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            print >> out, col,
        print >> out


def all_binary_permutations(n_bit):
    """
    used in brute forcing the partition function
    """
    rval = numpy.zeros((2 ** n_bit, n_bit))

    for i, val in enumerate(xrange(2 ** n_bit)):
        t = bin(val)[2:].zfill(n_bit)
        t = [int(s) for s in t]
        t = numpy.asarray(t)
        rval[i] = t

    return rval


def exp_linear_nonlinear_transformation():
    pass


def analyze_weights(path):
    # adjust k accordingly
    # usage: in the exp folder, run: RAB_tools.py model_params*.pkl
    params_path = sort_by_numbers_in_file_name(path)
    epoch_numbers = extract_epoch_number(params_path)
    # number of params
    k = 5
    boxes = [[], [], [], [], []]
    infos = [[], [], [], [], []]

    for epoch, path in enumerate(params_path):
        params = load_pkl(path)
        assert len(params) == len(boxes)
        for i, param in enumerate(params):
            boxes[i].append(abs(param).mean())
            infos[i] = 'shape:' + str(param.shape)
    fig = plt.figure(figsize=(15, 13))
    for i, box in enumerate(boxes):
        ax = plt.subplot2grid((k, 1), (i, 0))
        ax.plot(epoch_numbers, box, label=infos[i])
        ax.legend(loc=7, prop={'size': 15})
    plt.suptitle('change of mean(abs(param))', fontsize=20)
    plt.xlabel('training epoch', fontsize=15)
    plt.savefig('params_mag_change_all_epochs.png')
    plt.show()


def test_resample():
    x = numpy.asarray([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [1, 1], [1, 1],
                       [2, 2], [2, 2],
                       [3, 3], [3, 3],
                       [4, 4], [4, 4],
                       [5, 5], [5, 5]])
    y = numpy.asarray([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                       [1, 1], [1, 1],
                       [2, 2], [2, 2],
                       [3, 3], [3, 3],
                       [4, 4], [4, 4],
                       [5, 5], [5, 5]])

    new_x, new_y = resample(x, y)


def test_plot_tsne():
    numpy.load()


if __name__ == "__main__":
    # table = [["", "taste", "land speed", "life"],
    #         ["spam", 1.99, 4, 1003],
    #         ["eggs", 105, 13, 42],
    #         ["lumberjacks", 13, 105, 10]]

    # pprint_table(table)
    # plot_cost_from_npz()
    # divide_to_3_folds(150)
    # all_binary_permutations(n_bit=15)
    # plot_two_vector()
    # import ipdb; ipdb.set_trace()
    # t1 = ['jobman000/samples_1.png']
    # t2 = ['jobman000/samples_2.png']
    # t3 = ['jobman000/samples_3.png']
    # t4 = ['jobman000/samples_4.png']
    # t5 = ['jobman000/samples_5.png']
    # t = t1 + t2 + t4 + t3 + t5
    # sort_by_numbers_in_file_name(t)
    # import ipdb; ipdb.set_trace()
    # plot_noisy_tanh()

    # path = sys.argv[1:]
    # analyze_weights(path)

    # show_random_sampling_graph()
    # plot_learning_curves_from_npz()
    # test_resample()
    generate_geometric_sequence(1, 500, 2)
