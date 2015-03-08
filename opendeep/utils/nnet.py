"""
.. module:: nnet

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

As well as Pylearn2.utils (https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/utils)
and theano_alexnet (https://github.com/uoguelph-mlrg/theano_alexnet)
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import numpy
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal imports
from opendeep import cast32, trunc

log = logging.getLogger(__name__)

numpy.random.RandomState(23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

def get_shared_weights(n_in, n_out, interval=None, name="W"):
    if interval is None:
        interval = numpy.sqrt(6. / (n_in + n_out))
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_bias(n, name="b", offset=0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_hiddens(in_size, hidden_size, batch_size, i, name="H"):
    if i==0:
        val = numpy.zeros((batch_size,in_size))
    else:
        val = numpy.zeros((batch_size,hidden_size))
    return theano.shared(value=val,name=name)

def get_shared_regression_weights(hidden_size, name="V"):
#     val = numpy.identity(hidden_size)
    interval = numpy.sqrt(6. / (hidden_size + hidden_size))
    val = numpy.random.uniform(-interval, interval, size=(hidden_size, hidden_size))
    
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val


def dropout(IN, p = 0.5, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    OUT     =   (IN * noise) / cast32(p)
    return OUT

def add_gaussian_noise(IN, std = 1, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    log.debug('GAUSSIAN NOISE : %s', str(std))
    noise   =   MRG.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
    OUT     =   IN + noise
    return OUT

def corrupt_input(IN, p = 0.5, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    # salt and pepper? masking?
    noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    IN      =   IN * noise
    return IN

def salt_and_pepper(IN, p = 0.2, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    # salt and pepper noise
    a = MRG.binomial(size=IN.shape, n=1,
                          p = 1 - p,
                          dtype='float32')
    b = MRG.binomial(size=IN.shape, n=1,
                          p = 0.5,
                          dtype='float32')
    c = T.eq(a,0) * b
    return IN * a + c


def fix_input_size(xs, hiddens=None):
    # Make the dimensions of all X's in xs be the same. (with xs being a list of 2-dimensional matrices)
    sizes = [x.shape[0] for x in xs]
    min_size = numpy.min(sizes)
    xs = [x[:min_size] for x in xs]
    if hiddens is not None:
        hiddens = [hiddens[i][:min_size] for i in range(len(hiddens))]
    return xs, hiddens

def copy_params(params):
        values = [param.get_value(borrow=True) for param in params]
        return values

def restore_params(params, values):
    for i in range(len(params)):
        params[i].set_value(values[i])

def load_from_config(config_filename):
    log.debug('Loading local config file')
    config_file =   open(config_filename, 'r')
    config      =   config_file.readlines()
    try:
        config_vals =   config[0].split('(')[1:][0].split(')')[:-1][0].split(', ')
    except:
        config_vals =   config[0][3:-1].replace(': ','=').replace("'","").split(', ')
        config_vals =   filter(lambda x:not 'jobman' in x and not '/' in x and not ':' in x and not 'experiment' in x, config_vals)
    
    return config_vals