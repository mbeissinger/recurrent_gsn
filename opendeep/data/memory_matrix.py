'''
Dataset object wrapper for something given in memory (array of arrays, numpy matrix)
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import numpy
# internal imports
from opendeep.utils.utils import sharedX
import opendeep.data.dataset as datasets
from opendeep.data.dataset import Dataset
import opendeep.utils.file_ops as files
from opendeep.utils.utils import make_shared_variables

log = logging.getLogger(__name__)

class MemoryMatrix(Dataset):
    '''
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix)
    '''
    def __init__(self, train_X, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None):
        log.info('Wrapping matrix from memory')
        super(self.__class__, self).__init__()

        # make sure the inputs are arrays
        train_X = numpy.array(train_X)
        self._train_shape = train_X.shape
        self.train_X = sharedX(train_X)
        if train_Y:
            self.train_Y = sharedX(numpy.array(train_Y))

        if valid_X:
            valid_X = numpy.array(valid_X)
            self._valid_shape = valid_X.shape
            self.valid_X = sharedX(valid_X)
        if valid_Y:
            self.valid_Y = sharedX(numpy.array(valid_Y))

        if test_X:
            test_X = numpy.array(test_X)
            self._test_shape = test_X.shape
            self.test_X = sharedX(test_X)
        if test_Y:
            self.test_Y = sharedX(numpy.array(test_Y))

    def getDataByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset values at the index (indices)
        '''
        if subset is datasets.TRAIN:
            return self.train_X.get_value(borrow=True)[indices]
        elif subset is datasets.VALID and hasattr(self, 'valid_X') and self.valid_X:
            return self.valid_X.get_value(borrow=True)[indices]
        elif subset is datasets.TEST and hasattr(self, 'test_X') and self.test_X:
            return self.test_X.get_value(borrow=True)[indices]
        else:
            return None

    def getLabelsByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data label values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset labels at the index (indices)
        '''
        if subset is datasets.TRAIN and hasattr(self, 'train_Y') and self.train_Y:
            return self.train_Y.get_value(borrow=True)[indices]
        elif subset is datasets.VALID and hasattr(self, 'valid_Y') and self.valid_Y:
            return self.valid_Y.get_value(borrow=True)[indices]
        elif subset is datasets.TEST and hasattr(self, 'test_Y') and self.test_Y:
            return self.test_Y.get_value(borrow=True)[indices]
        else:
            return None