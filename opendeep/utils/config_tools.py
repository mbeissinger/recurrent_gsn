'''
Methods used for parsing various configurations into dictionary-like objects
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
import collections
import os
import json
# third-party libraries
import yaml

log = logging.getLogger(__name__)

def create_dictionary_like(_input):
    # check if it is a dictionary-like object (implements collections.Mapping)
    if isinstance(_input, collections.Mapping):
        return _input
    # otherwise, check if it is a filename to a .json or .yaml
    elif os.path.isfile(_input):
        _, extension = os.path.splitext(_input)
        # if ends in .json
        if extension.lower() is '.json':
            with open(_input, 'r') as json_data:
                return json.load(json_data)
        # if ends in .yaml
        elif extension.lower() is '.yaml':
            with open(_input, 'r') as yaml_data:
                return yaml.load(yaml_data)
        else:
            log.critical('Configuration file %s with extension %s not supported', str(_input), extension)
            return False
    # otherwise not recognized/supported:
    else:
        log.critical('Could not find config. Either was not collections.Mapping object or not found in filesystem.')
        return False


