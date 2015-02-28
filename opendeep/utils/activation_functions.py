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
# internal references
from opendeep import cast32

log = logging.getLogger(__name__)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def rectifier(x):
    return T.maximum(cast32(0), x)

def tanh(x):
    return T.tanh(x)

def get_activation_function(name):
    if name == 'sigmoid':
        return lambda x: sigmoid(x)
    elif name == 'rectifier':
        return lambda x: rectifier(x)
    elif name == 'tanh':
        return lambda x: tanh(x)
    else:
        log.critical("Did not recognize activation %s, please use tanh, rectifier, or sigmoid", str(name))
        raise NotImplementedError("Did not recognize activation {0!s}, please use tanh, rectifier, or sigmoid".format(name))