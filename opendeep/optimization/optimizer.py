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
__email__ = "dev@opendeep.com"

# standard libraries
import logging

log = logging.getLogger(__name__)

class Optimizer(object):
    '''
    Default interface for an optimizer implementation - to train a model on a dataset
    '''
    def __init__(self, model, dataset, config, rng=None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.rng = rng

    def train(self):
        log.critical("You need to implement a 'train' function to train the model on the dataset with respect to parameters! Optimizer is %s", str(type(self)))
        raise NotImplementedError()
