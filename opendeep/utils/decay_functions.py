'''
Functions used for decaying Theano parameters
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
# third party libraries
import numpy

log = logging.getLogger(__name__)

class DecayFunction(object):
    '''
    Interface for a parameter decay function
    '''
    def __init__(self, param):
        # make sure the parameter is a Theano shared variable
        if not hasattr(param, 'get_value'):
            log.error('Parameter doesn\'t have a get_value() function!')
        if not hasattr(param, 'set_value'):
            log.error('Parameter doesn\'t have a set_value() function!')
        assert hasattr(param, 'get_value')
        assert hasattr(param, 'set_value')

    def reduce(self):
        log.critical('Parameter decay function %s does not have a reduce method!', str(type(self)))
        raise NotImplementedError()

    def simulate(self, initial, reduction_factor, epoch):
        log.critical('Parameter decay function %s does not have a simulate method!', str(type(self)))
        raise NotImplementedError()


class Linear(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param)
        self.param = param
        self.param.set_value(initial)
        self.reduction_factor = reduction_factor

    def reduce(self):
        new_value = self.param.get_value() - self.reduction_factor
        self.param.set_value(numpy.max([0, new_value]))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value - reduction_factor*epoch
        return numpy.max([0, new_value])


class Exponential(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param)
        self.param = param
        self.param.set_value(initial)
        self.reduction_factor = reduction_factor

    def reduce(self):
        new_value = self.param.get_value()*self.reduction_factor
        self.param.set_value(new_value)

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value*pow(reduction_factor, epoch)
        return new_value


class Montreal(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param)
        self.param = param
        self.param.set_value(initial)
        self.initial = initial
        self.reduction_factor = reduction_factor
        self.epoch = 1

    def reduce(self):
        new_value = self.initial / (1 + self.reduction_factor*self.epoch)
        self.param.set_value(new_value)
        self.epoch += 1

    def simulate(self, initial, reduction_factor, epoch):
        new_value = initial / (1 + reduction_factor*epoch)
        return new_value
