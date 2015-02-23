'''
A Convolutional Neural Network model

Based on the code described by Sander Dieleman
http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
Which is an adaptation of Alex Krizhevsky's cuda-convnet
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
# third party libraries
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
# internal references
from opendeep.models.model import Model

log = logging.getLogger(__name__)

class ConvNet(Model):
    '''
    A fast convnet implementation
    '''
    def __init__(self):
        super(self.__class__, self).__init__()
        self.input = T.tensor4('input')
        self.filters = T.tensor4('filters')
