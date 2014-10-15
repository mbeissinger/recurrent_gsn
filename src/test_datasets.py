'''
Created on Sep 8, 2014

@author: markus
'''
import numpy
import data_tools as data
import theano
import PIL.Image
from image_tiler import tile_raster_images
import time
from utils import trunc, fix_input_size

def test(n, batch_size = 121, iters = 100, xslength=21):
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
    train_pre = len(train_Y.eval())
    valid_pre = len(valid_Y.eval())
    test_pre = len(test_Y.eval())
    print 'train set size:',train_pre
    print 'valid set size:',valid_pre
    print 'test set size:',test_pre
    print 
    for i in range(len(train_Y.get_value(borrow=True)) / batch_size):
        xs = [train_Y.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(xslength)]
        xs, _ = fix_input_size(xs)
        if i<10:
            print i
            print numpy.transpose(numpy.vstack([x[0] for x in xs]))
    print
    for i in range(len(valid_Y.get_value(borrow=True)) / batch_size):
        x = valid_Y.get_value()[i * batch_size : (i+1) * batch_size]
        x1 = valid_Y.get_value()[(i * batch_size) + 1 : ((i+1) * batch_size) + 1]
        [x,x1], _ = fix_input_size([x,x1])
        if i<5 or i==(len(valid_Y.get_value(borrow=True)) / batch_size)-1:
            if i==(len(valid_Y.get_value(borrow=True)) / batch_size)-1:
                print '...'
            print x[0],x1[0],"...",x[-1],x1[-1]
    print
    for i in range(len(test_Y.get_value(borrow=True)) / batch_size):
        x = test_Y.get_value()[i * batch_size : (i+1) * batch_size]
        x1 = test_Y.get_value()[(i * batch_size) + 1 : ((i+1) * batch_size) + 1]
        [x,x1], _ = fix_input_size([x,x1])
        if i<5 or i==(len(test_Y.get_value(borrow=True)) / batch_size)-1:
            if i==(len(test_Y.get_value(borrow=True)) / batch_size)-1:
                print '...'
            print x[0],x1[0],"...",x[-1],x1[-1]
    print
    print train_Y.get_value()[:80]
    print valid_Y.get_value()[:80]
    print test_Y.get_value()[:80]
    print
    print
    
    save_dataset(train_X, 'train_output_dataset_'+str(n)+'.png')
    save_dataset(valid_X, 'valid_output_dataset_'+str(n)+'.png')
    save_dataset(test_X, 'test_output_dataset_'+str(n)+'.png')
    
    print 'starting',iters,'iterations...',
    t = time.time()
    #run 1000 sequences to see if lengths change
    for _ in range(iters):
        data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, n)
    timing = time.time() - t
    print 'took',trunc(timing),'seconds'
    print 'difference in sizes after',iters,'iterations:'
    print 'train set:',train_pre - len(train_Y.eval())
    print 'valid set:',valid_pre - len(valid_Y.eval())
    print 'test set:',test_pre - len(test_Y.eval())
    print
    print
    
    
def save_dataset(dataset, name):
    N_input =   dataset.get_value().shape[1]
    root_N_input = numpy.sqrt(N_input)
    numbers = dataset.get_value()[0:100]
    stacked = numpy.vstack([numpy.vstack([numbers[i*10 : (i+1)*10]]) for i in range(10)])
    number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,10)))
    number_reconstruction.save(name)
    
def fix_size(xs):
    sizes = [x.shape for x in xs]
    min_size = numpy.min(sizes)
    xs = [x[:min_size] for x in xs]
    return xs


if __name__ == "__main__":
    #test(0)
    test(1)
    #test(2)
    #test(3)
