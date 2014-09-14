'''
Created on Sep 8, 2014

@author: markus
'''
import numpy
import data_tools as data
import theano
import PIL.Image
from image_tiler import tile_raster_images

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
    
    print_dataset(train_X, 'train_output_dataset_'+str(n)+'.png')
    print_dataset(valid_X, 'valid_output_dataset_'+str(n)+'.png')
    print_dataset(test_X, 'test_output_dataset_'+str(n)+'.png')
    
    
def print_dataset(dataset, name):
    N_input =   dataset.get_value().shape[1]
    root_N_input = numpy.sqrt(N_input)
    numbers = dataset.get_value()[0:100]
    stacked = numpy.vstack([numpy.vstack([numbers[i*10 : (i+1)*10]]) for i in range(10)])
    number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,10)))
    number_reconstruction.save(name)


if __name__ == "__main__":
    test(0)
    test(1)
    test(2)
    test(3)
