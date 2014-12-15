'''
Created on Jul 18, 2014

@author: markus
'''

import numpy, os, sys, cPickle
import numpy.random as rng
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from utils.image_tiler import *
import time
import argparse
from utils import data_tools as data
import random as R

cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))

def sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
    #shuffle the datasets
    train_indices = range(len(train_Y.get_value(borrow=True)))
    rng.shuffle(train_indices)
    valid_indices = range(len(valid_Y.get_value(borrow=True)))
    rng.shuffle(valid_indices)
    test_indices = range(len(test_Y.get_value(borrow=True)))
    rng.shuffle(test_indices)
    
    train_X.set_value(train_X.get_value(borrow=True)[train_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_indices])
    
    # Find the order of MNIST data going from 0-9 repeating
    train_ordered_indices = data.create_series(train_Y.get_value(borrow=True), 10)
    valid_ordered_indices = data.create_series(valid_Y.get_value(borrow=True), 10)
    test_ordered_indices = data.create_series(test_Y.get_value(borrow=True), 10)
    
    # Put the data sets in order
    train_X.set_value(train_X.get_value(borrow=True)[train_ordered_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_ordered_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_ordered_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_ordered_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_ordered_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_ordered_indices])
    
def plot(samples, root_N_input, name):
    img_samples =   PIL.Image.fromarray(tile_raster_images(samples, (root_N_input,root_N_input), (1,10)))
    fname       =   'average_mnist_{0!s}.png'.format(name)
    img_samples.save(fname)
    
    
    
def main():
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist('.')
    train_X = numpy.concatenate((train_X, valid_X))
    train_Y = numpy.concatenate((train_Y, valid_Y))
        
    train_X = theano.shared(train_X)
    train_Y = theano.shared(train_Y)
    valid_X = theano.shared(valid_X)
    valid_Y = theano.shared(valid_Y) 
    test_X = theano.shared(test_X)
    test_Y = theano.shared(test_Y) 
   
    sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
    
    N_input =   train_X.eval().shape[1]
    root_N_input = numpy.sqrt(N_input)
    
    train_size = train_X.eval().shape[0]
    valid_size = valid_X.eval().shape[0]
    test_size = test_X.eval().shape[0]

    # variables
    X = T.fmatrix('X')
    
    #functions
    avg_X = T.mean(X, axis=0)
    avg = theano.function(inputs=[X], outputs=avg_X)

    train = []
    valid = []
    test = []
    for i in range(10):
        train_indices = [j+i for j in xrange(0, train_size, 10) if j+i < train_size]
        valid_indices = [j+i for j in xrange(0, valid_size, 10) if j+i < valid_size]
        test_indices = [j+i for j in xrange(0, test_size, 10) if j+i < test_size]
        
        train.append(train_X.get_value()[train_indices])
        valid.append(valid_X.get_value()[valid_indices])
        test.append(test_X.get_value()[test_indices])
              
    avg_train = [avg(x) for x in train]
    avg_valid = [avg(x) for x in valid]
    avg_test = [avg(x) for x in test]
        
    plot(numpy.vstack(avg_train), root_N_input, "train")
    plot(numpy.vstack(avg_valid), root_N_input, "valid")
    plot(numpy.vstack(avg_test), root_N_input, "test")
    
    
    print 'getting cross entropies....'
    #find the average cross-entropy
    train_costs = []
    for i in range(10):
        size = train[i].shape[0]
        a = avg_train[i]
        xs = train[i]
        xs[xs==0] = 0.1
        xs[xs==1] = 0.9
        a[a==0] = 0.1
        a[a==1] = 0.9
        cost = numpy.mean(T.nnet.binary_crossentropy(xs, a).eval())
        print cost
        train_costs.append(cost)
            
    print 'train:',[trunc(c) for c in train_costs],"|",trunc(numpy.mean(train_costs))
    
    valid_costs = []
    for i in range(10):
        size = valid[i].shape[0]
        compare = numpy.vstack([avg_valid[i][0] for j in xrange(size)])
        cost = T.nnet.binary_crossentropy(output, compare).mean()
        valid_costs.append(cost)
            
    print 'valid:',[trunc(c) for c in valid_costs],"|",trunc(numpy.mean(valid_costs))
    
    test_costs = []
    for i in range(10):
        size = test[i].shape[0]
        compare = numpy.vstack([avg_test[i][0] for j in xrange(size)])
        cost = T.nnet.binary_crossentropy(output, compare).mean()
        test_costs.append(cost)
    
    print 'test:',[trunc(c) for c in test_costs],"|",trunc(numpy.mean(test_costs))



if __name__ == '__main__':
    main()