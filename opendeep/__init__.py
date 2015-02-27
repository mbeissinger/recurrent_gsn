'''
Code used throughout the entire OpenDeep package.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# third-party libraries
import theano
import numpy

cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))

def function(*args, **kwargs):
    """
    Taken from Pylearn2 https://github.com/lisa-lab/pylearn2

    A wrapper around theano.function that disables the on_unused_input error.
    Almost no part of OpenDeep can assume that an unused input is an error, so
    the default from theano is inappropriate for this project.
    """
    return theano.function(*args, on_unused_input='warn', **kwargs)

def grad(*args, **kwargs):
    """
    Taken from Pylearn2 https://github.com/lisa-lab/pylearn2

    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error. Almost no part of OpenDeep can assume that a disconnected input
    is an error.
    """
    return theano.gradient.grad(*args, disconnected_inputs='warn', **kwargs)