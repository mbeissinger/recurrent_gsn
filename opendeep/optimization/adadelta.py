'''
Generic structure of ADADELTA algorithm

'ADADELTA: An Adaptive Learning Rate Method'
Matthew D. Zeiler
http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
'''


# TODO: SHOULD THIS BE AN ADAPTIVE LEARNING RATE OBJECT FOR SGD LIKE PYLEARN2 DOES IT??


__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
import time
from collections import OrderedDict
# third party libraries
import numpy
import numpy.random as random
import theano
import theano.tensor as T
# internal references
from opendeep.optimization.optimizer import Optimizer
from opendeep.utils.decay_functions import get_decay_function
from opendeep.data.iterators.sequential import SequentialIterator
import opendeep.data.dataset as datasets
from opendeep.utils.utils import cast32, make_time_units_string, copy_params, restore_params

log = logging.getLogger(__name__)

# Default values to use for some training parameters
defaults = {}

class AdaDelta(Optimizer):
    '''
    The ADADELTA training algorithm
    '''
    def __init__(self):
        pass