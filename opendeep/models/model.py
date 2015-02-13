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

log = logging.getLogger(__name__)

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
        log.critical("The Model %s does not have a cost function!", str(type(self)))
        raise NotImplementedError()

    def get_output_function(self):
        log.critical("The Model %s does not have an output function!", str(type(self)))
        raise NotImplementedError()
