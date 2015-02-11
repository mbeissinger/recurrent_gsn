'''
Basic interface for an optimizer
'''

class Optimizer(object):
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError("You need to implement a 'train' function to train the model on the dataset with respect to parameters!")