'''
General interface for creating a model.
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
# third party libraries
import theano
import theano.tensor as T
# internal references
from opendeep.utils.utils import make_time_units_string

log = logging.getLogger(__name__)

# TODO: Deeplearning4J would call this interface 'layer' and then string them together for the given number of layers
# TODO: look into the ramifications for making it distributed.
# TODO: maybe make a 'model' object instantiate a bunch of 'layer' objects (or not, like the case of gsn)
class Model(object):
    '''
    Default interface for creating a model. Contains the barebones methods that should be implemented.
    '''

    def __init__(self, config=None):
        '''
        :param config: a dictionary-like object containing all the necessary parameters for the model. Could be parsed by JSON.
        '''
        self.config = config


    def get_inputs(self):
        log.critical("The Model %s does not have a get_inputs function!", str(type(self)))
        raise NotImplementedError()


    def get_updates(self):
        log.critical("The Model %s does not have a get_updates function!", str(type(self)))
        raise NotImplementedError()


    def get_outputs(self):
        log.critical("The Model %s does not have an output method!", str(type(self)))
        raise NotImplementedError()


    def get_cost(self):
        log.critical("The Model %s does not have a cost method!", str(type(self)))
        raise NotImplementedError()


    def get_monitors(self):
        '''
        :return: list
        A list of the theano variables to use as the monitors.
        '''
        log.critical("The Model %s does not have a monitors method!", str(type(self)))
        raise NotImplementedError()


    def get_monitor_function(self):
        if not hasattr(self, 'f_monitors'):
            log.info('Compiling f_monitor function...')
            t = time.time()
            self.f_monitors = theano.function(inputs  = self.get_inputs(),
                                              updates = self.get_updates(),
                                              outputs = self.get_monitors(),
                                              name    = 'f_monitors')
            log.info('f_monitor compilation took %s', make_time_units_string(time.time() - t))
        return self.f_monitors


    def load_params(self, param_file):
        log.critical("The Model %s does not have load_params!!", str(type(self)))
        raise NotImplementedError()


    def save_params(self, param_file_suffix):
        log.critical("The Model %s does not have save_params!", str(type(self)))
        raise NotImplementedError()


    # return a list of the parameter variables
    def get_params(self):
        raise NotImplementedError()


    # return a list of the parameter values
    def get_params_values(self):
        raise NotImplementedError()


    # return a list of the decay functions on internal parameters to decay each time step
    def get_decay_params(self):
        return []


    def score(self, X):
        """
        Compute a "score function" for this model, if this model has
        probabilistic semantics.
        Parameters
        ----------
        X : tensor_like, 2-dimensional
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
        log.critical("The Model %s does not have a probabilistic score method! (i.e. p(X=x|H))", str(type(self)))
        raise NotImplementedError()

