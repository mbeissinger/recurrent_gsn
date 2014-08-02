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

def get_google_data(dataset, rand, n_lines=1000000,most_common_flag = False):
#     train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
#train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    pos_map = get_pos_map()
    eigenword_map = {}
    print 'reading eigenwords'
    with open('../denoising/data/eigenwords_google.txt') as f:
        for line in f:
            split = line.split()
            eigenword_map[split[0]] = split[1:]
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''

    print '... loading data'

    # Load the dataset
    with open(dataset) as f:
        lines = f.readlines()
    print 'shuffling...'
    rand.shuffle(lines)
    if n_lines is not None:
        lines = lines[0:n_lines]
    total = len(lines)
    train_end =  np.int(total * .70)
    valid_end = np.int(total*.85)
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []
    print 'converting to eigenwords'
    pos_list = []
    for i in range(total):
        words = lines[i].split()
        x = []
        for word in words:
            x.extend([float(z) for z in eigenword_map.get(word)])
        pos = nltk.pos_tag(words)
        y = pos_map.get(pos[1][1])
        if y == None:
            y = float(get_output_size()-1)
        else:
            y = float(y)
        if i < train_end:
            train_x.append(x)
            train_y.append(y)
            if most_common_flag:
                pos_list.append(pos[1][1])
        elif i < valid_end:
            valid_x.append(x)
            valid_y.append(y)
            if most_common_flag:
                pos_list.append(pos[1][1])
        else:
            test_x.append(x)
            test_y.append(y)
        if i==train_end:
            print 'finished train set'
        elif i== valid_end:
            print 'finished validation set'
    print 'finished test set'
    train_set = (np.array(train_x), np.array(train_y))
    valid_set = (np.array(valid_x), np.array(valid_y))
    test_set = (np.array(test_x), np.array(test_y))

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    if not most_common_flag:
        return rval
    else:
        pos_tag_frequencies = nltk.FreqDist(pos_list)
        most_common = pos_tag_frequencies.max()
        return rval, float(pos_map.get(most_common))
    
#train_set, valid_set, test_set format: tuple(input, target)
#input is an numpy.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#numpy.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.
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

def get_pos_map():
    m = {}
    m["CC"]=0
    m["CD"]=1
    m["DT"]=2
    m["EX"]=3
    m["FW"]=4
    m["IN"]=5
    m["JJ"]=6
    m["JJR"]=7
    m["JJS"]=8
    m["LS"]=9
    m["MD"]=10
    m["NN"]=11
    m["NNS"]=12
    m["NNP"]=13
    m["NNPS"]=14
    m["PDT"]=15
    m["POS"]=16
    m["PRP"]=17
    m["PRP$"]=18
    m["RB"]=19
    m["RBR"]=20
    m["RBS"]=21
    m["RP"]=22
    m["SYM"]=23
    m["TO"]=24
    m["UH"]=25
    m["VB"]=26
    m["VBD"]=27
    m["VBG"]=28
    m["VBN"]=29
    m["VBP"]=30
    m["VBZ"]=31
    m["WDT"]=32
    m["WP"]=33
    m["WP$"]=34
    m["WRB"]=35
    m[":"]=36
    m["``"]=37
    m[","]=38
    m["#"]=39
    m["."]=40
    m["$"]=41
    m["-NONE-"]=42
    return m

def get_output_size():
    m = get_pos_map()
    return len(m.items())+1

def get_random_data(d_features=90,n_examples=10000,n_distributions=20):
    rng = np.random.RandomState(123)
    #define n_distribution random tuples of (mean, std) to define normal distributions
    #mean_std_list = [(float(rng.random_sample(1)[0]*200-100),float(rng.random_sample(1)[0]*.1+.001)) for i in range(n_distributions)]
    mean_std_list = [(rng.randint(-100,100),float(rng.random_sample(1)[0]*.1+.001)) for i in range(n_distributions)]
    #create the training X and Y
    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    test_X = []
    test_Y = []
    for i in range(n_examples):
        #randomly pick a distribution
        (mean,std) = choice(mean_std_list)
        #generate d_features from a normal distribution and add it to the list
        train_X.append([float(rng.normal(mean,std)) for d in range(d_features)])
        train_Y.append([mean, std])
        
    #create the validation X and Y
    for i in range(int(n_examples/2)):    
        #randomly pick a distribution
        (mean,std) = choice(mean_std_list)
        #generate d_features from a normal distribution and add it to the list
        valid_X.append([float(rng.normal(mean,std)) for d in range(d_features)])
        valid_Y.append([mean, std])
        
    #create the test X and Y
    for i in range(int(n_examples/2)):
        #randomly pick a distribution
        (mean,std) = choice(mean_std_list)
        #generate d_features from a normal distribution and add it to the list
        test_X.append([float(rng.normal(mean,std)) for d in range(d_features)])
        test_Y.append([mean, std])
        
    train_set = (np.array(train_X), np.array(train_Y))
    valid_set = (np.array(valid_X), np.array(valid_Y))
    test_set = (np.array(test_X), np.array(test_Y))

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def get_toy_data(d_features=784,n_examples=60000,n_distributions=10):
    rng = np.random.RandomState(123)
    #dists_list = [rng.randint(-100,100) for i in range(n_distributions)]
    dists_list = [[float(rng.random_sample(1)[0])]*d_features for i in range(n_distributions)]
    #dists_list = [[float(.1*n) if (d > 78*n and d < 78*(n+1)) else float(1-(.1*n)) for d in range(d_features)] for n in range(n_distributions)]
    #dists_list = [[float(1) if (d > 78*n and d < 78*(n+1)) else float(0) for d in range(d_features)] for n in range(n_distributions)]
    #create the training X and Y
    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    test_X = []
    test_Y = []
    for i in range(n_examples):
        #randomly pick a distribution
        (y,x) = choice(list(enumerate(dists_list)))
        #generate d_features from a normal distribution and add it to the list
        train_X.append(x)
        train_Y.append([y])
        
    #create the validation X and Y
    for i in range(int(n_examples/6)):    
        #randomly pick a distribution
        (y,x) = choice(list(enumerate(dists_list)))
        #generate d_features from a normal distribution and add it to the list
        valid_X.append(x)
        valid_Y.append([y])
        
    #create the test X and Y
    for i in range(int(n_examples/6)):
        #randomly pick a distribution
        (y,x) = choice(list(enumerate(dists_list)))
        #generate d_features from a normal distribution and add it to the list
        test_X.append(x)
        test_Y.append([y])
        
    train_set = (np.array(train_X), np.array(train_Y))
    valid_set = (np.array(valid_X), np.array(valid_Y))
    test_set = (np.array(test_X), np.array(test_Y))

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def get_gray_data(d_features=90,n_examples=10000,n_distributions=20):
    rng = np.random.RandomState(123)
    #dists_list = [rng.randint(-100,100) for i in range(n_distributions)]
    dists_list = [float(rng.random_sample(1)[0]) for i in range(n_distributions)]
    #create the training X and Y
    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    test_X = []
    test_Y = []
    for i in range(n_examples):
        #randomly pick a distribution
        mean = choice(dists_list)
        #generate d_features from a normal distribution and add it to the list
        train_X.append([float(mean) for d in range(d_features)])
        train_Y.append([mean])
        
    #create the validation X and Y
    for i in range(int(n_examples/2)):    
        #randomly pick a distribution
        mean = choice(dists_list)
        #generate d_features from a normal distribution and add it to the list
        valid_X.append([float(mean) for d in range(d_features)])
        valid_Y.append([mean])
        
    #create the test X and Y
    for i in range(int(n_examples/2)):
        #randomly pick a distribution
        mean = choice(dists_list)
        #generate d_features from a normal distribution and add it to the list
        test_X.append([float(mean) for d in range(d_features)])
        test_Y.append([mean])
        
    train_set = (np.array(train_X), np.array(train_Y))
    valid_set = (np.array(valid_X), np.array(valid_Y))
    test_set = (np.array(test_X), np.array(test_Y))

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

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
    
def sequence_data(X,Y,indices):
    sequenced_X = X[indices]
    sequenced_Y = Y[indices]
    return sequenced_X, sequenced_Y
    
    
    
    