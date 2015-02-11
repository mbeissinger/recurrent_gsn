'''
General interface for creating a model.
'''

class Model(object):
    '''
    Default class for creating a model. Contains the barebones methods that should be implemented.
    '''

    def __init__(self, config=None):
        '''
        :param config: dictionary-like object specifying the parameters of the model.
        '''