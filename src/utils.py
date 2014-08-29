'''
Created on Aug 18, 2014

@author: Markus
'''
import numpy
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG



cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))



def get_shared_weights(n_in, n_out, interval=None, name="W"):
    #val = numpy.random.normal(0, sigma_sqr, size=(n_in, n_out))
    if interval is None:
        interval = numpy.sqrt(6. / (n_in + n_out))
                              
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_bias(n, name="b", offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_hiddens(in_size, hidden_size, batch_size, i, name):
    if i==0:
        val = numpy.zeros((batch_size,in_size))
    else:
        val = numpy.zeros((batch_size,hidden_size))
    return theano.shared(value=val,name=name)

def get_shared_recurrent_weights(network_size, name="V"):
    val = numpy.identity(network_size)
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val





def dropout(IN, p = 0.5, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    OUT     =   (IN * noise) / cast32(p)
    return OUT

def add_gaussian_noise(IN, std = 1, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    print 'GAUSSIAN NOISE : ', std
    noise   =   MRG.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
    OUT     =   IN + noise
    return OUT

def corrupt_input(IN, p = 0.5, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    # salt and pepper? masking?
    noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    IN      =   IN * noise
    return IN

def salt_and_pepper(IN, p = 0.2, MRG=None):
    if MRG is None:
        MRG = RNG_MRG.MRG_RandomStreams(1)
    # salt and pepper noise
    print 'DAE uses salt and pepper noise'
    a = MRG.binomial(size=IN.shape, n=1,
                          p = 1 - p,
                          dtype='float32')
    b = MRG.binomial(size=IN.shape, n=1,
                          p = 0.5,
                          dtype='float32')
    c = T.eq(a,0) * b
    return IN * a + c


def fix_input_size(xs, hiddens=None):
    sizes = [x.shape[0] for x in xs]
    min_size = numpy.min(sizes)
    xs = [x[:min_size] for x in xs]
    if hiddens is not None:
        hiddens = [xs[0] if i==0 else hiddens[i][:min_size] for i in range(len(hiddens))]
    return xs, hiddens

def load_from_config(config_filename):
    print 'Loading local config file'
    config_file =   open(config_filename, 'r')
    config      =   config_file.readlines()
    try:
        config_vals =   config[0].split('(')[1:][0].split(')')[:-1][0].split(', ')
    except:
        config_vals =   config[0][3:-1].replace(': ','=').replace("'","").split(', ')
        config_vals =   filter(lambda x:not 'jobman' in x and not '/' in x and not ':' in x and not 'experiment' in x, config_vals)
    
    return config_vals
    