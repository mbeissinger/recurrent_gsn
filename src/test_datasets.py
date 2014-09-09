'''
Created on Sep 8, 2014

@author: markus
'''
import numpy
import data_tools as data
import theano

def test(n):
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist('../data/')
    train_X = numpy.concatenate((train_X, valid_X))
    train_Y = numpy.concatenate((train_Y, valid_Y))

    train_X = theano.shared(train_X)
    train_Y = theano.shared(train_Y)
    valid_X = theano.shared(valid_X)
    valid_Y = theano.shared(valid_Y) 
    test_X = theano.shared(test_X)
    test_Y = theano.shared(test_Y)
    
    print 'Dataset {!s} ------------------'.format(n)
    print
    print 'train set size:',len(train_Y.eval())
    print 'valid set size:',len(valid_Y.eval())
    print 'test set size:',len(test_Y.eval())
    print
    data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, n)
    print 'train set size:',len(train_Y.eval())
    print 'valid set size:',len(valid_Y.eval())
    print 'test set size:',len(test_Y.eval())
    print 
    print train_Y.get_value()[:80]
    print valid_Y.get_value()[:80]
    print test_Y.get_value()[:80]
    print
    print


if __name__ == "__main__":
    test(1)
    test(2)
    test(3)
