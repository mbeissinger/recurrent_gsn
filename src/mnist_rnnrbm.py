# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import sys
import time

import PIL.Image
import numpy
import numpy.random as rng
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import data_tools as data
from utils.image_tiler import *

# Don't use a python long as this don't work on 32 bits computers.
rng.seed(1)
rng_stream = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

cast32 = lambda x: numpy.cast['float32'](x)
trunc = lambda x: str(x)[:8]
logit = lambda p: numpy.log(p / (1 - p))
binarize = lambda x: cast32(x >= 0.5)
sigmoid = lambda x: cast32(1. / (1 + numpy.exp(-x)))


def shared_normal(num_rows, num_cols, interval=None, name=None):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    if interval is None:
        interval = numpy.sqrt(6. / (num_rows + num_cols)) / 3
    return theano.shared(
        value=numpy.random.normal(scale=interval, size=(num_rows, num_cols)).astype(theano.config.floatX), name=name)


def shared_uniform(num_rows, num_cols, name):
    return theano.shared(value=numpy.asarray(numpy.random.uniform(
        low=-numpy.sqrt(6. / (num_rows + num_cols)),
        high=numpy.sqrt(6. / (num_rows + num_cols)),
        size=(num_rows, num_cols)),
        dtype=theano.config.floatX), name=name)


def shared_zeros(size, name):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(value=numpy.zeros(size, dtype=theano.config.floatX), name=name)


def fix_input_size(xs, u0):
    sizes = [x.shape[0] for x in xs]
    min_size = numpy.min(sizes)
    xs = [x[:min_size] for x in xs]
    u0 = u0[:min_size]
    return xs, u0


def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a theano shared variable of type floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def build_rbm(v, W, bv, bh, k=50):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

v : Theano vector or matrix
  If a matrix, multiple chains will be run in parallel (batch).
W : Theano matrix
  Weight matrix of the RBM.
bv : Theano vector
  Visible bias vector of the RBM.
bh : Theano vector
  Hidden bias vector of the RBM.
k : scalar or Theano scalar
  Length of the Gibbs chain.

Return a (v_sample, cost, monitor, updates) tuple:

v_sample : Theano vector or matrix with the same shape as `v`
  Corresponds to the generated sample(s).
cost : Theano scalar
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng_stream.binomial(size=mean_h.shape, n=1, p=mean_h,
                                dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng_stream.binomial(size=mean_v.shape, n=1, p=mean_v,
                                dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]
    crossentropy = T.mean(T.nnet.binary_crossentropy(mean_v, v))

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, crossentropy, updates


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''Construct a symbolic RNN-RBM and initialize parameters.

n_visible : integer
  Number of visible units.
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.

Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
          updates_generate) tuple:

v : Theano matrix
  Symbolic variable holding an input sequence (used during training)
v_sample : Theano matrix
  Symbolic variable holding the negative particles for CD log-likelihood
  gradient estimation (used during training)
cost : Theano scalar
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM (used during training)
monitor : Theano scalar
  Frame-level pseudo-likelihood (useful for monitoring during training)
params : tuple of Theano shared variables
  The parameters of the model to be optimized during training.
updates_train : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  training function.
v_t : Theano matrix
  Symbolic variable holding a generated sequence (used during sampling)
updates_generate : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  generation function.'''

    W = shared_uniform(n_visible, n_hidden, "W")
    bv = shared_zeros(n_visible, "bv")
    bh = shared_zeros(n_hidden, "bh")
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001, "Wuh")
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001, "Wuv")
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001, "Wvu")
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001, "Wuu")
    bu = shared_zeros(n_hidden_recurrent, "bu")

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared
    # variables

    v = T.fmatrix("v")  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden

    # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                              bh_t, k=50)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)

    v_sample, cost, monitor, crossentropy, updates_rbm = build_rbm(v, W, bv_t, bh_t, k=15)

    v_prediction, _, _, _, updates_predict = build_rbm(v[:-1], W, bv_t[1:], bh_t[1:], k=15)

    mse = T.mean(T.sqr(v_sample[1:] - v_prediction), axis=0)
    accuracy = T.mean(mse)

    updates_train.update(updates_rbm)
    updates_train.update(updates_predict)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=400)

    return (v, v_sample, cost, monitor, accuracy, crossentropy, params, updates_train, v_t,
            updates_generate)


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
sequences.'''

    def __init__(self, n_visible=784, n_hidden=1000, n_hidden_recurrent=100, lr=0.01, annealing=.99):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate'''

        decay = 0.95

        self.lr = theano.shared(value=lr, name="lr")
        self.annealing = annealing
        self.root_N_input = 28

        (v, v_sample, cost, monitor, accuracy, crossentropy, params, updates_train, v_t, updates_generate) = \
            build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent)

        updates_test = updates_train

        gradient = T.grad(cost, params, consider_constant=[v_sample])

        updates_train.update(((p, p - lr * g) for p, g in zip(params, gradient)))

        print
        'compiling functions...'
        print
        'train'
        self.train_function = theano.function(inputs=[v], outputs=[accuracy, monitor, crossentropy],
                                              updates=updates_train)
        print
        'test'
        self.test_function = theano.function(inputs=[v], outputs=[accuracy, crossentropy], updates=updates_test)
        print
        'generate'
        self.generate_function = theano.function(inputs=[], outputs=v_t,
                                                 updates=updates_generate)

        print
        'functions done.'
        print

    def train(self, batch_size=100, num_epochs=300):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
files converted to piano-rolls.

files : list of strings
  List of MIDI files that will be loaded as piano-rolls for training.
batch_size : integer
  Training sequences will be split into subsequences of at most this size
  before applying the SGD updates.
num_epochs : integer
  Number of epochs (pass over the training set) performed. The user can
  safely interrupt training with Ctrl+C at any time.'''

        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist("../datasets/")
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))

        print
        'Sequencing MNIST data...'
        print
        'train set size:', train_X.shape
        print
        'valid set size:', valid_X.shape
        print
        'test set size:', test_X.shape

        train_X = theano.shared(train_X)
        train_Y = theano.shared(train_Y)
        valid_X = theano.shared(valid_X)
        valid_Y = theano.shared(valid_Y)
        test_X = theano.shared(test_X)
        test_Y = theano.shared(test_Y)

        data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset=4)

        print
        'train set size:', train_X.shape.eval()
        print
        'valid set size:', valid_X.shape.eval()
        print
        'test set size:', test_X.shape.eval()
        print
        'Sequencing done.'
        print

        N_input = train_X.eval().shape[1]
        self.root_N_input = numpy.sqrt(N_input)

        times = []

        try:
            for epoch in xrange(num_epochs):
                t = time.time()
                print
                'Epoch %i/%i : ' % (epoch + 1, num_epochs)
                # sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
                accuracy = []
                costs = []
                crossentropy = []
                tests = []
                test_acc = []

                for i in range(len(train_X.get_value(borrow=True)) / batch_size):
                    t0=time.time()
                    xs = train_X.get_value(borrow=True)[(i * batch_size): ((i + 1) * batch_size)]
                    acc, cost, cross = self.train_function(xs)
                    accuracy.append(acc)
                    costs.append(cost)
                    crossentropy.append(cross)
                    print time.time()-t0
                print 'Train',numpy.mean(accuracy), 'cost', numpy.mean(costs), 'cross', numpy.mean(crossentropy),
                    
                for i in range(len(test_X.get_value(borrow=True)) / batch_size):
                    xs = train_X.get_value(borrow=True)[(i * batch_size): ((i + 1) * batch_size)]
                    acc, cost = self.test_function(xs)
                    test_acc.append(acc)
                    tests.append(cost)

                print
                '\t Test_acc', numpy.mean(test_acc), "cross", numpy.mean(tests)

                timing = time.time() - t
                times.append(timing)
                print
                'time : ', trunc(timing),
                print
                'remaining: ', (num_epochs - (epoch + 1)) * numpy.mean(times) / 60 / 60, 'hrs'
                sys.stdout.flush()

                # new learning rate
                new_lr = self.lr.get_value() * self.annealing
                self.lr.set_value(new_lr)

        except KeyboardInterrupt:
            print
            'Interrupted by user.'

    def generate(self, filename):
        '''Generate a sample sequence, plot the resulting piano-roll and save
it as a MIDI file.

filename : string
  A MIDI file will be created at this location.
show : boolean
  If True, a piano-roll of the generated sequence will be shown.'''

        gen = self.generate_function()

        img_samples = PIL.Image.fromarray(tile_raster_images(gen, (self.root_N_input, self.root_N_input), (20, 20)))

        img_samples.save(filename)


def test_rnnrbm(batch_size=100, num_epochs=300):
    model = RnnRbm()
    model.train(batch_size=batch_size, num_epochs=num_epochs)
    return model


if __name__ == '__main__':
    model = test_rnnrbm()
    model.generate('rnnrbm_generated.png')
