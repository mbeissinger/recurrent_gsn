'''
Functions used for costs
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T

log = logging.getLogger(__name__)

def binary_crossentropy(x, y):
    cost = T.mean(T.nnet.binary_crossentropy(x, y))
    return cost

def crossentropy(x, y):
    cost = T.mean(T.nnet.categorical_crossentropy(x, y))
    return cost

def square(x, y):
    #cost = T.log(T.mean(T.sqr(x-y)))
    #cost = T.log(T.sum(T.pow((x-y),2)))
    cost = T.mean(T.sqr(x-y))
    return cost

def pseudo_log(x, y):
    eps = 1e-6
    cost = T.sum(T.xlogx.xlogy0(x, y+eps) + T.xlogx.xlogy0(1-x, 1-y+eps)) / x.shape[0]
    return cost

def get_cost_function(name):
    name = name.lower()
    if name == 'binary_crossentropy':
        return lambda x, y: binary_crossentropy(x, y)
    elif name == 'crossentropy':
        return lambda x, y: crossentropy(x, y)
    elif name == 'square':
        return lambda x, y: square(x, y)
    elif name == 'pseudo_log':
        return lambda y, x: pseudo_log(x, y)
    else:
        log.critical("Did not recognize cost function %s, please use binary_crossentropy, crossentropy, square, or pseudo_log", name)
        raise NotImplementedError("Did not recognize cost function {0!s}. Please use binary_crossentropy, crossentropy, square, or pseudo_log".format(name))