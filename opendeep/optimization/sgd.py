'''
Generic stochastic gradient descent optimization
'''
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
defaults = {"cost_function": 'binary_crossentropy',
            "n_epoch": 1000,
            "batch_size": 100,
            "minimum_batch_size": 1,
            "save_frequency": 10,
            "early_stop_threshold": .9995,
            "early_stop_length": 30,
            "learning_rate": 0.25,
            "lr_decay": "exponential",
            "lr_factor": .995,
            "momentum": 0.5,
            "iterator": SequentialIterator}

class SGD(Optimizer):
    '''
    Stochastic gradient descent for training a model - includes early stopping
    '''
    #TODO: add conjugate gradient?

    def __init__(self, model, dataset, config=dict(), rng=None):
        # grab parameters from the config if it exists, otherwise use the defaults
        # Training epochs - how many times to iterate over the whole dataset
        self.n_epoch = config.get('n_epoch', defaults['n_epoch'])
        # Dataset iteration batch sizes - number of examples in each calculation
        self.batch_size = config.get('batch_size', defaults['batch_size'])
        self.minimum_batch_size = config.get('minimum_batch_size', defaults['minimum_batch_size'])
        # Number of epochs between saving model parameters
        self.save_frequency = config.get('save_frequency', defaults['save_frequency'])
        # Early stopping threshold and patience - by how much does the cost have to improve over a number of epochs
        self.early_stop_threshold = config.get('early_stop_threshold', defaults['early_stop_threshold'])
        self.early_stop_length = config.get('early_stop_length', defaults['early_stop_length'])
        # Learning rate - how drastic of a step do the parameters change
        self.learning_rate = theano.shared(config.get('learning_rate', defaults['learning_rate']))
        self.learning_rate_decay = get_decay_function(config.get('lr_decay', defaults['lr_decay']),
                                                      self.learning_rate,
                                                      self.learning_rate.get_value(),
                                                      config.get('lr_factor', defaults['lr_factor']))
        # Momentum - smoothing over the parameter changes
        self.momentum = config.get('momentum', defaults['momentum'])
        # Iterator - what class of dataset iterator to use
        self.iterator = config.get('iterator', defaults['iterator'])

        self.model = model
        self.params = self.model.get_params()
        self.dataset = dataset

        # RNG for working on random iterator
        if rng is None:
            random.seed(123)
            self.rng = random
        else:
            self.rng = rng

        # Now create the training cost function for the model to use while training - update parameters
        log.info("%s params: %s", str(type(self.model)), str(self.params))
        # Stochastic gradient descent!
        gradient        =   T.grad(self.model.get_cost(), self.params)
        gradient_buffer =   [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in self.params]
        m_gradient      =   [self.momentum * gb + (cast32(1) - self.momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
        param_updates   =   [(param, param - self.learning_rate * mg) for (param, mg) in zip(self.params, m_gradient)]
        gradient_buffer_updates = zip(gradient_buffer, m_gradient)
        gradient_updates    =   OrderedDict(param_updates + gradient_buffer_updates)

        train_updates = model.get_updates()
        train_updates.update(gradient_updates)

        # Compile the functions!
        self.f_learn = theano.function(inputs  = model.get_inputs(),
                                       updates = train_updates,
                                       outputs = self.model.get_cost(),
                                       name    = 'f_learn',
                                       on_unused_input= 'warn')

        self.f_cost  = theano.function(inputs  = model.get_inputs(),
                                       updates = model.get_updates(),
                                       outputs = self.model.get_cost(),
                                       name    = 'f_cost',
                                       on_unused_input= 'warn')


    def train(self, continue_training=False):
        log.info("-----------TRAINING %s FOR %s EPOCHS (continue_training=%s)-----------", str(type(self.model)), str(self.n_epoch), str(continue_training))
        STOP    = False
        counter = 0
        if not continue_training:
            # reset the learning rate
            self.learning_rate_decay.reset()
            # reset the other model decaying functions
            for decay_param in self.model.get_decay_params():
                decay_param.reset()

        times       = []
        best_cost   = float('inf')
        best_params = None
        patience    = 0

        start_time = time.time()

        while not STOP:
            counter += 1
            t = time.time()
            # log.info(self.logger, [counter,'\t'])

            #train
            train_costs = []
            for x, y in self.iterator(self.dataset, datasets.TRAIN, self.batch_size, self.minimum_batch_size, self.rng):
                train_cost = self.f_learn([x, y])
                train_costs.append(train_cost)
            # log.maybeAppend(self.logger, ['Train:',trunc(numpy.mean(train_costs)), '\t'])

            #valid
            if self.dataset.hasSubset(datasets.VALID):
                valid_costs = []
                for x, y in self.iterator(self.dataset, datasets.VALID, self.batch_size, self.minimum_batch_size, self.rng):
                    valid_cost = self.f_cost([x, y])
                    valid_costs.append(valid_cost)
                # log.maybeAppend(self.logger, ['Valid:',trunc(numpy.mean(valid_costs)), '\t'])

            #test
            if self.dataset.hasSubset(datasets.TEST):
                test_costs = []
                for x, y in self.iterator(self.dataset, datasets.TEST, self.batch_size, self.minimum_batch_size, self.rng):
                    test_cost = self.f_cost([x, y])
                    test_costs.append(test_cost)
                # log.maybeAppend(self.logger, ['Test:',trunc(numpy.mean(test_costs)), '\t'])

            #check for early stopping
            if self.dataset.hasSubset(datasets.VALID):
                # use the first monitor for the cost checking
                cost = numpy.sum(valid_costs)
            else:
                cost = numpy.sum(train_costs)
            if cost < best_cost*self.early_stop_threshold:
                patience = 0
                best_cost = cost
                # save the parameters that made it the best
                best_params = copy_params(self.params)
            else:
                patience += 1

            if counter >= self.n_epoch or patience >= self.early_stop_length:
                STOP = True
                if best_params is not None:
                    restore_params(self.params, best_params)

            timing = time.time() - t
            times.append(timing)

            # log.maybeAppend(self.logger, 'time: '+make_time_units_string(timing)+'\t')

            # log.maybeLog(self.logger, 'remaining: '+make_time_units_string((self.n_epoch - counter) * numpy.mean(times)))

            if (counter % self.save_frequency) == 0 or STOP is True:
                #save params
                self.model.save_params('_epoch_'+str(counter))

            # ANNEAL!
            self.learning_rate_decay.decay()
            for decay_param in self.model.get_decay_params():
                decay_param.decay()

        log.info("------------TOTAL %s SGD TRAIN TIME TOOK %s---------", str(type(self.model)), make_time_units_string(time.time()-start_time))