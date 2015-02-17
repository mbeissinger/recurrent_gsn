'''
Basic interface for an optimizer
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

class Optimizer(object):
    '''
    Default interface for an optimizer implementation - to train a model on a dataset
    '''
    def __init__(self, model, dataset, config, rng=None):
        pass

    def train(self):
        log.critical("You need to implement a 'train' function to train the model on the dataset with respect to parameters! Optimizer is %s", str(type(self)))
        raise NotImplementedError()
