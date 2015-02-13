'''
General interface for creating a dataset iterator object.
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

class Iterator(object):
    '''
    Default interface for a Dataset iterator
    '''
    def __init__(self):
        pass


