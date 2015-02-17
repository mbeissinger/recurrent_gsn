'''
Object for the Piano-midi.de midi dataset
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

class PianoMidiDe(Dataset):
    '''
    Object for the Piano-midi.de midi dataset. Pickled file of midi piano roll provided by Montreal's Nicolas Boulanger-Lewandowski into train, valid, and test sets.
    '''
    def __init__(self, filename='Piano-midi.de.zip', source='http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.zip'):
        super(self.__class__, self).__init__(filename, source)
