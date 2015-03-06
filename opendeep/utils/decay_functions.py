'''
Functions used for decaying Theano parameters
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
import numpy
# internal references
from opendeep import cast32

log = logging.getLogger(__name__)

class DecayFunction(object):
    '''
    Interface for a parameter decay function (like learning rate, noise levels, etc.)
    '''
    def __init__(self, param, initial, reduction_factor):
        # make sure the parameter is a Theano shared variable
        if not hasattr(param, 'get_value'):
            log.error('Parameter doesn\'t have a get_value() function!')
        if not hasattr(param, 'set_value'):
            log.error('Parameter doesn\'t have a set_value() function!')
        assert hasattr(param, 'get_value')
        assert hasattr(param, 'set_value')
        self.param = param
        self.initial = initial
        self.param.set_value(cast32(self.initial))
        self.reduction_factor = reduction_factor

    def decay(self):
        log.critical('Parameter decay function %s does not have a decay method!', str(type(self)))
        raise NotImplementedError()

    def reset(self):
        self.param.set_value(self.initial)

    def simulate(self, initial, reduction_factor, epoch):
        log.critical('Parameter decay function %s does not have a simulate method!', str(type(self)))
        raise NotImplementedError()


class Linear(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        if self.reduction_factor is None:
            self.reduction_factor = 0

    def decay(self):
        new_value = self.param.get_value() - self.reduction_factor
        self.param.set_value(cast32(numpy.max([0, new_value])))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value - reduction_factor*epoch
        return numpy.max([0, new_value])


class Exponential(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        if self.reduction_factor is None:
            self.reduction_factor = 1

    def decay(self):
        new_value = self.param.get_value()*self.reduction_factor
        self.param.set_value(cast32(new_value))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value*pow(reduction_factor, epoch)
        return new_value


class Montreal(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        self.epoch = 1
        if self.reduction_factor is None:
            self.reduction_factor = 0

    def decay(self):
        new_value = self.initial / (1 + self.reduction_factor*self.epoch)
        self.param.set_value(cast32(new_value))
        self.epoch += 1

    def simulate(self, initial, reduction_factor, epoch):
        new_value = initial / (1 + reduction_factor*epoch)
        return new_value


def get_decay_function(name, parameter, initial, reduction_factor):
    name = name.lower()
    if name == 'linear':
        return Linear(parameter, initial, reduction_factor)
    elif name == 'exponential':
        return Exponential(parameter, initial, reduction_factor)
    elif name == 'montreal':
        return Montreal(parameter, initial, reduction_factor)
    else:
        log.critical("Did not recognize decay function %s, please use linear, exponential, or montreal", name)
        raise NotImplementedError("Did not recognize cost function {0!s}, please use linear, exponential, or montreal".format(name))