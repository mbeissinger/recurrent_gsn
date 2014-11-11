'''
Created on Nov 2, 2013

@author: markus
'''
import numpy
import theano
import theano.tensor as T
import os, cPickle
import gzip
import errno
from utils import cast32
import urllib
import scipy.io as io


#create a filesystem path if it doesn't exist.
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
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX), # @UndefinedVariable
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


#downloads the mnist data to specified file
def download_mnist(path):
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    gzip_file = os.path.join(path,'mnist.pkl.gz')
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, gzip_file)

def load_mnist(path):
    ''' Loads the mnist dataset

    :type path: string
    :param dataset: the path to the directory containing MNIST
    '''
    mkdir_p(path)
    
    pkl_file = os.path.join(path,'mnist.pkl')
    gzip_file = os.path.join(path,'mnist.pkl.gz')
    
    #check if a pkl or gz file exists
    if os.path.isfile(pkl_file):
        data = cPickle.load(open(pkl_file, 'r'))
    elif os.path.isfile(gzip_file):
        data = cPickle.load(gzip.open(gzip_file, 'rb'))
    else: #otherwise, it doesn't exist - download from lisa lab
        download_mnist(path)
        # Load the dataset
        data = cPickle.load(gzip.open(gzip_file, 'rb'))
    
    return data

def load_mnist_binary(path):
    pkl_file = os.path.join(path,'mnist.pkl')
    gzip_file = os.path.join(path,'mnist.pkl.gz')
    
    #check if a pkl or gz file exists
    if os.path.isfile(pkl_file):
        data = cPickle.load(open(pkl_file, 'r'))
    elif os.path.isfile(gzip_file):
        data = cPickle.load(gzip.open(gzip_file, 'rb'))
    else: #otherwise, it doesn't exist - download from lisa lab
        download_mnist(path)
        # Load the dataset
        data = cPickle.load(gzip.open(gzip_file, 'rb'))
    
    #make binary
    data = [list(d) for d in data] 
    data[0][0] = (data[0][0] > 0.5).astype('float32')
    data[1][0] = (data[1][0] > 0.5).astype('float32')
    data[2][0] = (data[2][0] > 0.5).astype('float32')
    data = tuple([tuple(d) for d in data])
    return data
    
def load_tfd(path):
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


        
def dataset1_indices(labels, classes=10):
    #Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset) that makes the numbers go in order 0-9....
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print
            print "stopped early from dataset1 sequencing - missing some class of labels"
            print
    while not stop:
        #for i in range(classes)+range(classes-2,0,-1):
        for i in range(classes):
            if not stop:
                if len(pool[i]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence
        
#order sequentially, but randomly choose when a 1, 4, or 8.
def dataset2_indices(labels, rng, classes=10, change_prob=.5):
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print "stopped early from dataset2 sequencing - missing some class of labels"
    while not stop:
        for i in range(classes):
            if not stop:
                n = i
                if i==1:
                    if rng.sample() < change_prob:
                        n = rng.choice([4,8])
                if i==4:
                    if rng.sample() < change_prob:
                        n = rng.choice([1,8])
                elif i==8:
                    if rng.sample() < change_prob:
                        n = rng.choice([4,1])
                if len(pool[n]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
    return sequence

#order sequentially up then down
def dataset2a_indices(labels, classes=10):
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print
            print "stopped early from dataset2a sequencing - missing some class of labels"
            print
    while not stop:
        for i in range(classes)+range(classes-1,-1,-1):
            if not stop:
                if len(pool[i]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

#order sequentially up then down but only for 0-1-1-0-0-1.....
def dataset2b_indices(labels, classes=2):
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        if labels[i] < len(pool):
            pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print
            print "stopped early from dataset2b sequencing - missing some class of labels"
            print
    while not stop:
        for i in range(classes)+range(classes-1,-1,-1):
            if not stop:
                if len(pool[i]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence
                

def dataset3_indices(labels, classes=10):
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print "stopped early from dataset3 sequencing - missing some class of labels"
    a = False
    while not stop:
        for i in range(classes):
            if not stop:
                n=i
                if i == 1 and a:
                    n = 4
                elif i == 4 and a:
                    n = 8
                elif i == 8 and a:
                    n = 1
                if len(pool[n]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
        a = not a
            
    return sequence


def sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset=1, rng=None, one_hot=False):
    if rng is None:
        rng = numpy.random
        rng.seed(1)
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
    
    # Find the order of MNIST data going from 0-9 repeating if the first dataset        
    if dataset == 1:
        train_ordered_indices = dataset1_indices(train_Y.get_value(borrow=True))
        valid_ordered_indices = dataset1_indices(valid_Y.get_value(borrow=True))
        test_ordered_indices = dataset1_indices(test_Y.get_value(borrow=True))
    elif dataset == 2:
        train_ordered_indices = dataset2a_indices(train_Y.get_value(borrow=True))
        valid_ordered_indices = dataset2a_indices(valid_Y.get_value(borrow=True))
        test_ordered_indices = dataset2a_indices(test_Y.get_value(borrow=True))
    elif dataset == 3:
        train_ordered_indices = dataset3_indices(train_Y.get_value(borrow=True))
        valid_ordered_indices = dataset3_indices(valid_Y.get_value(borrow=True))
        test_ordered_indices = dataset3_indices(test_Y.get_value(borrow=True))
    else:
        train_ordered_indices = train_indices
        valid_ordered_indices = valid_indices
        test_ordered_indices = test_indices
    
    # Put the data sets in order
    train_X.set_value(train_X.get_value(borrow=True)[train_ordered_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_ordered_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_ordered_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_ordered_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_ordered_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_ordered_indices])
    
    
    ###############################################################################################################
    # For testing one-hot encoding to see if it can learn sequences without having to worry about nonlinear input #
    ###############################################################################################################
    if one_hot:
        #construct the numpy matrix of representations from y
        train = numpy.array([[1 if i==y else 0 for i in range(10)] for y in train_Y.get_value(borrow=True)],dtype="float32")
        train_X.set_value(train)
        valid = numpy.array([[1 if i==y else 0 for i in range(10)] for y in valid_Y.get_value(borrow=True)],dtype="float32")
        valid_X.set_value(valid)
        test = numpy.array([[1 if i==y else 0 for i in range(10)] for y in test_Y.get_value(borrow=True)],dtype="float32")
        test_X.set_value(test)
    
    
    
    
    