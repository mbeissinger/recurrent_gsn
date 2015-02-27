'''
Object for the Nottingham midi dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# internal imports
from dataset import Dataset

log = logging.getLogger(__name__)

class Nottingham(Dataset):
    '''
    Object for the Nottingham midi dataset. Pickled file of midi piano roll provided by Montreal's Nicolas Boulanger-Lewandowski into train, valid, and test sets.
    '''
    def __init__(self, filename='Nottingham.zip', source='http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip'):
        super(self.__class__, self).__init__(filename, source)
