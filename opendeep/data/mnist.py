'''
Object for the MNIST handwritten digit dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
# internal imports
from dataset import Dataset

log = logging.getLogger(__name__)

class MNIST(Dataset):
    '''
    Object for the MNIST handwritten digit dataset. Pickled file provided by Montreal's LISA lab into train, valid, and test sets.
    '''
    def __init__(self, filename='mnist.pkl.gz', source='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'):
        super(self.__class__, self).__init__(filename, source)


    def iterator(self, mode=None, batch_size=None, minimum_batch_size=None, rng=None):
        pass