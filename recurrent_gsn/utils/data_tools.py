'''
@author: Markus Beissinger
University of Pennsylvania, 2014-2015

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN
'''
import numpy
import theano
import theano.tensor as T
import os, cPickle, gzip, errno, urllib, glob, zipfile
from utils import cast32
import scipy.io as io
from midi.utils import midiread

# Define the re-used loops for f_learn and f_cost
def apply_cost_function_to_dataset(function, dataset, batch_size=1):
    costs = []
    for i in xrange(len(dataset.get_value(borrow=True)) / batch_size):
        xs = dataset.get_value(borrow=True)[i * batch_size : (i+1) * batch_size]
#         xs = dataset[i * batch_size : (i+1) * batch_size].eval()
        cost, error = function(xs)
        costs.append([cost, error])
    return costs

def apply_indexed_cost_function_to_dataset(function, dataset_length, batch_size=1):
    costs = []
    for i in xrange(dataset_length / batch_size):
        cost = function(i)
        costs.append(cost)
    return costs

#create a filesystem path if it doesn't exist.
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def download_file(origin, destination_path, filename):
    print 'Downloading data from %s' % origin
    gzip_file = os.path.join(destination_path, filename)
    urllib.urlretrieve(origin, gzip_file)
    
def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)

#downloads the mnist data to specified file
def download_mnist(path):
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    download_file(origin, path, filename)
    
def download_piano_midi_de(path):
    origin = 'http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.zip'
    filename = 'Piano-midi.de.zip'
    download_file(origin, path, filename)
    unzip(os.path.join(path, filename), path)

def download_nottingham(path):
    origin = 'http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip'
    filename = 'Nottingham.zip'
    download_file(origin, path, filename)
    unzip(os.path.join(path, filename), path)
    
def download_muse(path):
    origin = 'http://www-etud.iro.umontreal.ca/~boulanni/MuseData.zip'
    filename = 'MuseData.zip'
    download_file(origin, path, filename)
    unzip(os.path.join(path, filename), path)
    
def download_jsb(path):
    origin = 'http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.zip'
    filename = 'JSB Chorales.zip'
    download_file(origin, path, filename)
    unzip(os.path.join(path, filename), path)

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

def load_piano_midi_de(path):
    mkdir_p(path)
    d = os.path.join(path,'Piano-midi.de')
    if not os.path.isdir(d):
        download_piano_midi_de(path)
    
    train_filenames = os.path.join(path, 'Piano-midi.de', 'train', '*.mid')
    valid_filenames = os.path.join(path, 'Piano-midi.de', 'valid', '*.mid')
    test_filenames = os.path.join(path, 'Piano-midi.de', 'test', '*.mid')
    train_files = glob.glob(train_filenames)
    valid_files = glob.glob(valid_filenames)
    test_files = glob.glob(test_filenames)
    
    train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
    valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
    test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
    
    return (train_datasets,[None]), (valid_datasets,[None]), (test_datasets,[None])
    

def load_nottingham(path):
    mkdir_p(path)
    d = os.path.join(path,'Nottingham')
    if not os.path.isdir(d):
        download_nottingham(path)
    
    train_filenames = os.path.join(path, 'Nottingham', 'train', '*.mid')
    valid_filenames = os.path.join(path, 'Nottingham', 'valid', '*.mid')
    test_filenames = os.path.join(path, 'Nottingham', 'test', '*.mid')
    train_files = glob.glob(train_filenames)
    valid_files = glob.glob(valid_filenames)
    test_files = glob.glob(test_filenames)
    
    train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
    valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
    test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
    
    return (train_datasets,[None]), (valid_datasets,[None]), (test_datasets,[None])

def load_muse(path):
    mkdir_p(path)
    d = os.path.join(path,'MuseData')
    if not os.path.isdir(d):
        download_muse(path)
    
    train_filenames = os.path.join(path, 'MuseData', 'train', '*.mid')
    valid_filenames = os.path.join(path, 'MuseData', 'valid', '*.mid')
    test_filenames = os.path.join(path, 'MuseData', 'test', '*.mid')
    train_files = glob.glob(train_filenames)
    valid_files = glob.glob(valid_filenames)
    test_files = glob.glob(test_filenames)
    
    train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
    valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
    test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
    
    return (train_datasets,[None]), (valid_datasets,[None]), (test_datasets,[None])

def load_jsb(path):
    mkdir_p(path)
    d = os.path.join(path,'JSB Chorales')
    if not os.path.isdir(d):
        download_jsb(path)
    
    train_filenames = os.path.join(path, 'JSB Chorales', 'train', '*.mid')
    valid_filenames = os.path.join(path, 'JSB Chorales', 'valid', '*.mid')
    test_filenames = os.path.join(path, 'JSB Chorales', 'test', '*.mid')
    train_files = glob.glob(train_filenames)
    valid_files = glob.glob(valid_filenames)
    test_files = glob.glob(test_filenames)
    
    train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
    valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
    test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
    
    return (train_datasets,[None]), (valid_datasets,[None]), (test_datasets,[None])
    
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



def load_datasets(dataset, data_path):
    """
    Load the appropriate dataset as (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) tuples.

    @type  dataset: String
    @param dataset: Name of the dataset to return.
    @type  data_path: String
    @param data_path: Location of the data directory to use.
    
    @rtype:  Tuples
    @return: (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)
    """
    dataset = dataset.lower()
    if dataset.startswith("mnist"):
        if dataset == "mnist_binary":
            return load_mnist_binary(data_path)
        else:
            (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist(data_path)
            try:
                dataset = int(dataset.split('_')[1])
                return sequence_mnist_not_shared(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset)
            except:
                return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)
    elif dataset == "tfd":
        return load_tfd(data_path)
    elif dataset == "nottingham":
        return load_nottingham(data_path)
    elif dataset == "muse":
        return load_muse(data_path)
    elif dataset == "pianomidi":
        return load_piano_midi_de(data_path)
    elif dataset == "jsb":
        return load_jsb(data_path)
    else:
        raise NotImplementedError("You requested to load dataset {0!s}, please choose MNIST*, TFD, nottingham, muse, pianomidi, jsb.".format(dataset))


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    # Unpack the x and y data
    data_x, data_y = data_xy
#     shared_x = theano.shared(numpy.asarray(data_x,
#                                            dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)
#     shared_y = theano.shared(numpy.asarray(data_y,
#                                            dtype=theano.config.floatX), # @UndefinedVariable
#                              borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
#     return shared_x, T.cast(shared_y, 'int32')
    
    return theano.shared(data_x, borrow=borrow), theano.shared(data_y, borrow=borrow)


def shuffle_data(X, Y=None, rng=None):
    if X is None:
        pass
    if rng is None:
        rng = numpy.random
        rng.seed(1)
    #shuffle the dataset, making sure to keep X and Y together
    train_indices = range(len(X.get_value(borrow=True)))
    rng.shuffle(train_indices)
    
    X.set_value(X.get_value(borrow=True)[train_indices], borrow=True)
    if Y is not None:
        Y.set_value(Y.get_value(borrow=True)[train_indices], borrow=True)

    

        
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

#order sequentially up then down 0-9-9-0....
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

# extra bits of parity
def dataset4_indices(labels, classes=10):
    def even(n):
        return n%2==0
    def odd(n):
        return not even(n)
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
            print "stopped early from dataset4 sequencing - missing some class of labels"
    s = [0,1,2]
    sequence.append(pool[0].pop())
    sequence.append(pool[1].pop())
    sequence.append(pool[2].pop())
    while not stop:
        if odd(s[-3]):
            first_bit = (s[-2] - s[-3])%classes
        else:
            first_bit = (s[-2] + s[-3])%classes
        if odd(first_bit):
            second_bit = (s[-1] - first_bit)%classes
        else:
            second_bit = (s[-1] + first_bit)%classes
        if odd(second_bit):
            next_num = second_bit = (s[-1] - second_bit)%classes
        else:
            next_num = second_bit = (s[-1] + second_bit + 1)%classes 
            
        if len(pool[next_num]) == 0: #stop the procedure if you are trying to pop from an empty list
            stop = True
        else:
            s.append(next_num)
            sequence.append(pool[next_num].pop())
            
    return sequence


def sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset=1, rng=None, one_hot=False):
    if rng is None:
        rng = numpy.random
        rng.seed(1)
        
    def set_xy_indices(x, y, indices):
        x.set_value(x.get_value(borrow=True)[indices])
        y.set_value(y.get_value(borrow=True)[indices])
        
    def shuffle_xy(x, y):
        if x is not None and y is not None:
            indices = range(len(y.get_value(borrow=True)))
            rng.shuffle(indices)
            set_xy_indices(x, y, indices)
        
    #shuffle the datasets
#     shuffle_xy(train_X, train_Y)
#     shuffle_xy(valid_X, valid_Y)
#     shuffle_xy(test_X, test_Y)
    
    # Find the order of MNIST data going from 0-9 repeating if the first dataset
    order_i = True
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
    elif dataset == 4:
        train_ordered_indices = dataset4_indices(train_Y.get_value(borrow=True))
        valid_ordered_indices = dataset4_indices(valid_Y.get_value(borrow=True))
        test_ordered_indices = dataset4_indices(test_Y.get_value(borrow=True))
    else:
        order_i = False
    
    # Put the data sets in order
    if order_i:
        set_xy_indices(train_X, train_Y, train_ordered_indices)
        set_xy_indices(valid_X, valid_Y, valid_ordered_indices)
        set_xy_indices(test_X, test_Y, test_ordered_indices)
    
    
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
    
def sequence_mnist_not_shared(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset=1):        
    def set_xy_indices(x, y, indices):
        x = x[indices]
        y = y[indices]
        return x,y
   
    # Find the order of MNIST data going from 0-9 repeating if the first dataset
    order_i = True
    if dataset == 1:
        train_ordered_indices = dataset1_indices(train_Y)
        valid_ordered_indices = dataset1_indices(valid_Y)
        test_ordered_indices = dataset1_indices(test_Y)
    elif dataset == 2:
        train_ordered_indices = dataset2a_indices(train_Y)
        valid_ordered_indices = dataset2a_indices(valid_Y)
        test_ordered_indices = dataset2a_indices(test_Y)
    elif dataset == 3:
        train_ordered_indices = dataset3_indices(train_Y)
        valid_ordered_indices = dataset3_indices(valid_Y)
        test_ordered_indices = dataset3_indices(test_Y)
    elif dataset == 4:
        train_ordered_indices = dataset4_indices(train_Y)
        valid_ordered_indices = dataset4_indices(valid_Y)
        test_ordered_indices = dataset4_indices(test_Y)
    else:
        order_i = False
    
    # Put the data sets in order
    if order_i:
        train_X, train_Y = set_xy_indices(train_X, train_Y, train_ordered_indices)
        valid_X, valid_Y = set_xy_indices(valid_X, valid_Y, valid_ordered_indices)
        test_X, test_Y   = set_xy_indices(test_X, test_Y, test_ordered_indices)
        
    return (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)
        
    
        
        
        
        
           
    
    