'''
Created on Nov 2, 2013

@author: markus
'''
import nltk
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from random import choice
import os, sys, cPickle
import gzip
import errno
from utils import cast32

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_mnist(path):
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    return data

def load_mnist_binary(path):
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    data = [list(d) for d in data] 
    data[0][0] = (data[0][0] > 0.5).astype('float32')
    data[1][0] = (data[1][0] > 0.5).astype('float32')
    data[2][0] = (data[2][0] > 0.5).astype('float32')
    data = tuple([tuple(d) for d in data])
    return data
    
def load_tfd(path):
    import scipy.io as io
    data = io.loadmat(os.path.join(path, 'TFD_48x48.mat'))
    X = cast32(data['images'])/cast32(255)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    labels  = data['labs_ex'].flatten()
    labeled = labels != -1
    unlabeled   =   labels == -1  
    train_X =   X[unlabeled]
    valid_X =   X[unlabeled][:100] # Stuf
    test_X  =   X[labeled]

    del data

    return (train_X, labels[unlabeled]), (valid_X, labels[unlabeled][:100]), (test_X, labels[labeled])

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval






def create_series(labels, classes):
    #Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset) that makes the numbers go in order 0-9....
    seen = range(classes)
    #Initiate a list of indices with the worst case size: #input labels * #possible classes
    indices = [-1]*(labels.shape[0]*classes)
    e = labels
    for i in range(len(e)):
        for c in range(classes):
            if e[i] == c:
                indices[seen[c]] = i
                seen[c] += classes
                    
    end_idx = np.where(np.array(indices) == -1)[0]
    if end_idx.size == 0:        
        return indices
    else:
        if end_idx[0] == 0:
            raise Exception('missing first class from sequence labels')
        else:
            return indices[0:end_idx[0]]
    

def sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset=1, rng=None):
    if rng is None:
        rng = np.random.seed(1)
    #shuffle the datasets
    train_indices = range(len(train_Y.get_value(borrow=True)))
    rng.shuffle(train_indices)
    valid_indices = range(len(valid_Y.get_value(borrow=True)))
    rng.shuffle(valid_indices)
    test_indices = range(len(test_Y.get_value(borrow=True)))
    rng.shuffle(test_indices)
    
    train_X.set_value(train_X.get_value(borrow=True)[train_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_indices])
    
    # Find the order of MNIST data going from 0-9 repeating
    train_ordered_indices = create_series(train_Y.get_value(borrow=True), 10)
    valid_ordered_indices = create_series(valid_Y.get_value(borrow=True), 10)
    test_ordered_indices = create_series(test_Y.get_value(borrow=True), 10)
    
    # Put the data sets in order
    train_X.set_value(train_X.get_value(borrow=True)[train_ordered_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_ordered_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_ordered_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_ordered_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_ordered_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_ordered_indices])
    
    
    
    