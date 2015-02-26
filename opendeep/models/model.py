'''
General interface for creating a model - this is the same for simple layers, modules, and a full-blown model.
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
# internal references
from opendeep import function
from opendeep.utils.config_tools import create_dictionary_like
from opendeep.utils.utils import make_time_units_string
from opendeep.optimization.optimizer import Optimizer
from opendeep.optimization.stochastic_gradient_descent import SGD

log = logging.getLogger(__name__)

class Model(object):
    '''
    Default interface for creating a model (or any sub-component of one). Contains the barebones methods that should be implemented.
    Think of it like legos.
    '''

    def __init__(self, config=None, defaults=None):
        '''
        :param config: a dictionary-like object containing all the necessary parameters for the model. Could be a dictionary-like object, .json file, or .yaml file.
        :param defaults: a dictionary-like object containing all the necessary default parameters for the model. Could be a dictionary-like object, .json file, or .yaml file.
        '''
        self.config = create_dictionary_like(config)
        self.defaults = create_dictionary_like(defaults)

    def get_input_variables(self):
        # This should return the symbolic variable(s) that are the inputs to this model.
        # Used whenever a theano function for this model is created.
        log.critical("The Model %s does not have a get_input_variables function!", str(type(self)))
        raise NotImplementedError()

    def get_updates(self):
        # This should return any theano updates from the model (used for things like rng's). Most often comes from theano's 'scan' op. Check out documentation.
        # This is used with the optimizer to create the training function - the 'updates' part of the theano function.

        # log.critical("The Model %s does not have a get_updates function!", str(type(self)))
        # raise NotImplementedError()

        # by default, assume the model doesn't have updates - it is the job of the creator to return them in this method.
        return None

    def get_output_variables(self):
        # Returns the variable(s) that are the outputs of the model.
        # Used for hooking up simple layers/modules/models together. These output variables become the input to the next component.
        log.critical("The Model %s does not have a get_output_variables method!", str(type(self)))
        raise NotImplementedError()

    def get_cost(self):
        # Return the cost expression for this model. Used with the optimizer for training.
        log.critical("The Model %s does not have a get_cost method!", str(type(self)))
        raise NotImplementedError()

    def get_monitors(self):
        '''
        :return: list
        A list of the theano variables to use as the monitors during training - the model variables you want to check periodically.
        '''
        log.critical("The Model %s does not have a monitors method!", str(type(self)))
        raise NotImplementedError()

    def get_monitor_function(self):
        # Creates a theano function returning the value of the monitor variables. This function will be called during the optimizer's train method to output monitor values.
        if not hasattr(self, 'f_monitors'):
            log.info('Compiling f_monitor function for model %s...', str(type(self)))
            t = time.time()
            self.f_monitors = function(inputs  = self.get_input_variables(),
                                       updates = self.get_updates(),
                                       outputs = self.get_monitors(),
                                       name    = 'f_monitors')
            log.info('f_monitor compilation took %s', make_time_units_string(time.time() - t))
        return self.f_monitors

    def load_params(self, param_file):
        # for loading saved parameters for this model.
        log.critical("The Model %s does not have load_params!!", str(type(self)))
        raise NotImplementedError()

    def save_params(self, param_file):
        # for saving the parameters for this model.
        log.critical("The Model %s does not have save_params!", str(type(self)))
        raise NotImplementedError()

    def get_train_params(self):
        # return a list of the parameter variables. Used with the optimizer for training these params.
        log.critical("The Model %s does not have get_train_params!", str(type(self)))
        raise NotImplementedError()

    def get_decay_params(self):
        # return a list of the decay functions on internal parameters to decay each time step. Used with the optimizer.
        return []

    def score(self, X):
        """
        (description from pylearn2)
        Compute a "score function" for this model, if this model has
        probabilistic semantics.
        Parameters
        ----------
        X : tensor_like
            A batch of i.i.d. examples with examples indexed along the
            first axis and features along the second. This is data on which
            the monitoring quantities will be calculated (e.g., a validation
            set).
        Returns
        -------
        score : tensor_like
            The gradient of the negative log probability of the model
            on the given data.
        Notes
        -----
        If the model implements a probability distribution on R^n,
        this method should return the gradient of the log probability
        of the batch with respect to V, or raise an exception explaining
        why this is not possible.
        """
        log.critical("The Model %s does not have a probabilistic score method! (i.e. p(X=x|H)). This is used for log-likelihoods.",
                     str(type(self)))
        raise NotImplementedError()

    def train(self, dataset, optimizer=None, optimizer_config=None, rng=None):
        # use stochastic gradient descent by default
        if optimizer is None:
            optimizer = SGD

        assert isinstance(optimizer, Optimizer), "Model %s optimizer during train() is not an instance of Optimizer" % str(type(self))
        assert callable(optimizer), "Model %s optimizer during train() is not a callable class" % str(type(self))
        optimizer = optimizer(self, dataset, optimizer_config, rng)

        optimizer.train()

