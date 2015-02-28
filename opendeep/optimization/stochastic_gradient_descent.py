'''
Generic stochastic gradient descent optimization with momentum and annealing
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
import time
# third party libraries
import numpy
import numpy.random as random
from theano.compat.python2x import OrderedDict  # use this compatability OrderedDict
# internal references
from opendeep import function, grad, trunc
from opendeep.optimization.optimizer import Optimizer
from opendeep.utils.decay_functions import get_decay_function
from opendeep.data.iterators.sequential import SequentialIterator
import opendeep.data.dataset as datasets
from opendeep.utils.utils import cast32, make_time_units_string, copy_params, restore_params, sharedX

log = logging.getLogger(__name__)

# Default values to use for some training parameters
_defaults = {"n_epoch": 1000,
             "batch_size": 100,
             "minimum_batch_size": 1,
             "save_frequency": 10,
             "early_stop_threshold": .9995,
             "early_stop_length": 30,
             "learning_rate": 0.25,
             "lr_decay": "exponential",
             "lr_factor": .995,
             "momentum": 0.5,
             "unsupervised": False}

class SGD(Optimizer):
    '''
    Stochastic gradient descent for training a model - includes early stopping, momentum, and annealing
    '''

    def __init__(self, model, dataset, iteratorClass=SequentialIterator, config=dict(), defaults=_defaults, rng=None):
        super(self.__class__, self).__init__(model, dataset, iteratorClass, config, defaults, rng)
        # grab parameters from the config if it exists, otherwise use the defaults
        # Training epochs - how many times to iterate over the whole dataset
        self.n_epoch = self.args.get('n_epoch')

        # Dataset iteration batch sizes - number of examples in each calculation
        self.batch_size         = self.args.get('batch_size')
        self.minimum_batch_size = self.args.get('minimum_batch_size')

        # Number of epochs between saving model parameters
        self.save_frequency = self.args.get('save_frequency')

        # Early stopping threshold and patience - by how much does the cost have to improve over a number of epochs
        self.early_stop_threshold = self.args.get('early_stop_threshold')
        self.early_stop_length    = self.args.get('early_stop_length')

        # Learning rate - how drastic of a step do the parameters change
        self.learning_rate       = sharedX(cast32(self.args.get('learning_rate')), 'learning_rate')
        self.learning_rate_decay = get_decay_function(self.args.get('lr_decay'),
                                                      self.learning_rate,
                                                      self.learning_rate.get_value(),
                                                      self.args.get('lr_factor'))

        # Momentum - smoothing over the parameter changes (see Hinton)
        self.momentum = cast32(self.args.get('momentum'))

        self.params = self.model.get_params()

        self.unsupervised = self.args.get("unsupervised")

        # RNG for working on random iterator
        if rng is None:
            random.seed(123)
            self.rng = random
        else:
            self.rng = rng

        # Now create the training cost function for the model to use while training - update parameters
        log.info("%s params: %s", str(type(self.model)), str(self.params))
        # Stochastic gradient descent!
        gradient        = grad(self.model.get_train_cost(), self.params)
        gradient_buffer = [sharedX(numpy.zeros(param.get_value().shape, dtype='float32')) for param in self.params]
        m_gradient      = [self.momentum * gb + (cast32(1) - self.momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
        param_updates   = [(param, param - self.learning_rate * mg) for (param, mg) in zip(self.params, m_gradient)]
        gradient_buffer_updates = zip(gradient_buffer, m_gradient)
        gradient_updates= OrderedDict(param_updates + gradient_buffer_updates)

        train_updates = model.get_updates()
        if train_updates:
            train_updates.update(gradient_updates)
        else:
            train_updates = gradient_updates

        # Compile the functions!
        log.info('Compiling f_learn function for model %s...', str(type(self.model)))
        t = time.time()
        self.f_learn = function(inputs  = model.get_inputs(),
                                updates = train_updates,
                                outputs = self.model.get_train_cost(),
                                name    = 'f_learn')
        log.info('f_learn compilation took %s', make_time_units_string(time.time() - t))

        self.monitor_function = self.model.get_monitor_function()


    def train(self, continue_training=False):
        log.info("-----------TRAINING %s FOR %s EPOCHS (continue_training=%s)-----------", str(type(self.model)), str(self.n_epoch), str(continue_training))
        self.STOP    = False
        self.epoch_counter = 0
        if not continue_training:
            # reset the learning rate
            self.learning_rate_decay.reset()
            # reset the other model decaying functions
            for decay_param in self.model.get_decay_params():
                decay_param.reset()

        self.times       = []
        self.best_cost   = float('inf')
        self.best_params = None
        self.patience    = 0

        start_time = time.time()

        while not self.STOP:
            self.STOP = self._perform_one_epoch()

        log.info("------------TOTAL %s SGD TRAIN TIME TOOK %s---------", str(type(self.model)), make_time_units_string(time.time()-start_time))


    def _perform_one_epoch(self):
            self.epoch_counter += 1
            t = time.time()
            log.info('EPOCH %s', str(self.epoch_counter))

            #train
            train_costs = []
            train_monitors = []
            for x, y in self.iterator(self.dataset, datasets.TRAIN, self.batch_size, self.minimum_batch_size, self.rng):
                if self.unsupervised:
                    train_costs.append(self.f_learn(x))
                    train_monitors.append(self.monitor_function(x))
                else:
                    train_costs.append(self.f_learn(x, y))
                    train_monitors.append(self.monitor_function(x, y))
            log.info('Train: %s', trunc(numpy.mean(train_costs, 0)))
            log.info('Train monitors: %s', str([trunc(num) for num in numpy.mean(train_monitors, 0)]))

            #valid
            if self.dataset.hasSubset(datasets.VALID):
                valid_monitors = []
                for x, y in self.iterator(self.dataset, datasets.VALID, self.batch_size, self.minimum_batch_size, self.rng):
                    if self.unsupervised:
                        valid_monitors.append(self.monitor_function(x))
                    else:
                        valid_monitors.append(self.monitor_function(x, y))
                log.info('Valid monitors: %s', str([trunc(num) for num in numpy.mean(valid_monitors, 0)]))

            #test
            if self.dataset.hasSubset(datasets.TEST):
                test_monitors = []
                for x, y in self.iterator(self.dataset, datasets.TEST, self.batch_size, self.minimum_batch_size, self.rng):
                    if self.unsupervised:
                        test_monitors.append(self.monitor_function(x))
                    else:
                        test_monitors.append(self.monitor_function(x, y))
                log.info('Test monitors: %s', str([trunc(num) for num in numpy.mean(test_monitors, 0)]))

            #check for early stopping
            if self.dataset.hasSubset(datasets.VALID):
                # use the first monitor for the cost checking
                cost = numpy.sum(valid_monitors, 0)[0]
            else:
                cost = numpy.sum(train_costs)
            if cost < self.best_cost*self.early_stop_threshold:
                self.patience = 0
                self.best_cost = cost
                # save the parameters that made it the best
                self.best_params = copy_params(self.params)
            else:
                self.patience += 1

            if self.epoch_counter >= self.n_epoch or self.patience >= self.early_stop_length:
                self.STOP = True
                if self.best_params is not None:
                    restore_params(self.params, self.best_params)

            timing = time.time() - t
            self.times.append(timing)

            log.info('time: '+make_time_units_string(timing))

            log.info('remaining time: '+make_time_units_string((self.n_epoch - self.epoch_counter) * numpy.mean(self.times)))

            if (self.epoch_counter % self.save_frequency) == 0 or self.STOP is True:
                #save params
                self.model.save_params('_epoch_'+str(self.epoch_counter))

            # ANNEAL!
            self.learning_rate_decay.decay()
            for decay_param in self.model.get_decay_params():
                decay_param.decay()