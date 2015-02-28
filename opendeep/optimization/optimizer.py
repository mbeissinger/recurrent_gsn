'''
Basic interface for an optimizer

Some information from Andrej Karpath:
'In my own experience, Adagrad/Adadelta are "safer" because they don't depend so strongly on setting of learning rates
(with Adadelta being slightly better), but well-tuned SGD+Momentum almost always converges faster and at better final
values.' http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html

Also see:
'Practical recommendations for gradient-based training of deep architectures'
Yoshua Bengio
http://arxiv.org/abs/1206.5533

'No More Pesky Learning Rates'
Tom Schaul, Sixin Zhang, Yann LeCun
http://arxiv.org/abs/1206.1106
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# internal references
from opendeep.utils.config_tools import create_dictionary_like

log = logging.getLogger(__name__)

class Optimizer(object):
    '''
    Default interface for an optimizer implementation - to train a model on a dataset
    '''
    def __init__(self, model, dataset, iteratorClass, config, defaults, rng=None):
        # make sure the config is like a dictionary
        config_dict = create_dictionary_like(config)
        # make sure the defaults is like a dictionary
        defaults_dict = create_dictionary_like(defaults)
        # override any default values with the config (after making sure they parsed correctly)
        if config_dict and defaults_dict:
            defaults_dict.update(config_dict)
        # set this update combination to the model arguments
        self.args = defaults_dict

        log.debug("ARGS: %s", str(self.args))

        self.model = model
        self.dataset = dataset
        self.iterator = iteratorClass
        self.rng = rng

    def train(self):
        log.critical("You need to implement a 'train' function to train the model on the dataset with respect to parameters! Optimizer is %s", str(type(self)))
        raise NotImplementedError()
