"""
Functions used for decaying variables
"""
import keras.backend as K
# third party libraries
import numpy


class DecayFunction(object):
    """
    Interface for a parameter decay function
    """

    def __init__(self, param, initial, reduction_factor):
        self.param = param
        self.initial = initial
        K.set_value(self.param, self.initial)
        self.reduction_factor = reduction_factor

    def decay(self):
        raise NotImplementedError('Decay function {!s} does not have a decay method!'.format(type(self)))

    def reset(self):
        K.set_value(self.param, self.initial)

    def simulate(self, initial, reduction_factor, epoch):
        raise NotImplementedError('Decay function {!s} does not have a simulate method!'.format(type(self)))


class Linear(DecayFunction):
    def decay(self):
        new_value = K.eval(self.param) - self.reduction_factor
        K.set_value(self.param, numpy.max([0, new_value]))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value - reduction_factor * epoch
        return numpy.max([0, new_value])


class Exponential(DecayFunction):
    def decay(self):
        new_value = K.eval(self.param) * self.reduction_factor
        K.set_value(self.param, new_value)

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value * pow(reduction_factor, epoch)
        return new_value


class Montreal(DecayFunction):
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        self.epoch = 1

    def decay(self):
        new_value = self.initial / (1 + self.reduction_factor * self.epoch)
        K.set_value(self.param, new_value)
        self.epoch += 1

    def simulate(self, initial, reduction_factor, epoch):
        new_value = initial / (1 + reduction_factor * epoch)
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
        raise NotImplementedError(
            "Did not recognize cost function {!s}, please use linear, exponential, or montreal".format(name))
