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
# internal references
from opendeep.optimization.sgd import SGD

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
        pass

    def get_learn_function(self):
        log.critical("The Model %s does not have a learn function!", str(type(self)))
        raise NotImplementedError()

    def get_output_function(self):
        log.critical("The Model %s does not have an output function!", str(type(self)))
        raise NotImplementedError()

    def train(self, dataset, optimizer=SGD()):
        log.critical("The Model %s does not have a train method!", str(type(self)))
        raise NotImplementedError()


    def predict(self, dataset):
        log.critical("The Model %s does not have a predict method!", str(type(self)))
        raise NotImplementedError()


    def load_params(self, param_file):
        log.critical("The Model %s does not have load_params!!", str(type(self)))
        raise NotImplementedError()


    def save_params(self, param_file):
        log.critical("The Model %s does not have save_params!", str(type(self)))
        raise NotImplementedError()


    # return a list of the parameter variables
    def get_params(self):
        raise NotImplementedError()


    # return a list of the parameter values
    def get_params_values(self):
        raise NotImplementedError()


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

