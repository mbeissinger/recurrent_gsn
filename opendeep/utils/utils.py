'''
@author: Markus Beissinger
University of Pennsylvania, 2014-2015

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

As well as Pylearn2.utils
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

def make_shared_variables(variable_list, borrow=True):
    return (theano.shared(variable, borrow=borrow) for variable in variable_list)

def get_shared_weights(n_in, n_out, interval=None, name="W"):
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

def get_shared_hiddens(in_size, hidden_size, batch_size, i, name="H"):
    if i==0:
        val = numpy.zeros((batch_size,in_size))
    else:
        val = numpy.zeros((batch_size,hidden_size))
    return theano.shared(value=val,name=name)

def get_shared_regression_weights(hidden_size, name="V"):
#     val = numpy.identity(hidden_size)
    interval = numpy.sqrt(6. / (hidden_size + hidden_size))
    val = numpy.random.uniform(-interval, interval, size=(hidden_size, hidden_size))
    
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val
    
def get_activation_function(name):
    if name == 'sigmoid':
        return T.nnet.sigmoid
    elif name == 'rectifier':
        return lambda x : T.maximum(cast32(0), x)
    elif name == 'tanh':
        return lambda x : T.tanh(x)
    else:
        raise NotImplementedError("Did not recognize activation {0!s}, please use tanh, rectifier, or sigmoid".format(name))

def get_cost_function(name):
    eps = 1e-6
    if name == 'binary_crossentropy':
        return lambda x,y: T.mean(T.nnet.binary_crossentropy(x,y))
    elif name == 'square':
        #return lambda x,y: T.log(T.mean(T.sqr(x-y)))
        #return lambda x,y: T.log(T.sum(T.pow((x-y),2)))
        return lambda x,y: T.mean(T.sqr(x-y))
    elif name == 'pseudo_log':
        return lambda y,x: T.sum(T.xlogx.xlogy0(x, y+eps) + T.xlogx.xlogy0(1-x, 1-y+eps)) / x.shape[0]
    else:
        raise NotImplementedError("Did not recognize cost function {0!s}, please use binary_crossentropy, square, or pseudo_log".format(name))


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
    a = MRG.binomial(size=IN.shape, n=1,
                          p = 1 - p,
                          dtype='float32')
    b = MRG.binomial(size=IN.shape, n=1,
                          p = 0.5,
                          dtype='float32')
    c = T.eq(a,0) * b
    return IN * a + c


def fix_input_size(xs, hiddens=None):
    # Make the dimensions of all X's in xs be the same. (with xs being a list of 2-dimensional matrices)
    sizes = [x.shape[0] for x in xs]
    min_size = numpy.min(sizes)
    xs = [x[:min_size] for x in xs]
    if hiddens is not None:
        hiddens = [hiddens[i][:min_size] for i in range(len(hiddens))]
    return xs, hiddens

def copy_params(params):
        values = [param.get_value(borrow=True) for param in params]
        return values

def restore_params(params, values):
    for i in range(len(params)):
        params[i].set_value(values[i])

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

def init_empty_file(filename):
    with open(filename, 'w') as f:
        f.write("")
        
def make_time_units_string(time):
    # Show the time with appropriate easy-to-read units.
    if time < 0:
        return trunc(time*1000)+" milliseconds"
    elif time < 60:
        return trunc(time)+" seconds"
    elif time < 3600:
        return trunc(time/60)+" minutes"
    else:
        return trunc(time/3600)+" hours"
    
def raise_to_list(_input):
    if _input is None:
        return None
    elif isinstance(_input, list):
        return _input
    else:
        return [_input]
    
def stack_and_shared(_input):
    if _input is None:
        return None
    elif isinstance(_input, list):
        shared_ins = []
        for _in in _input:
            try:
                shared_ins.append(theano.shared(_in))
            except TypeError as _:
                shared_ins.append(_in)
        return T.stack(shared_ins)
    else:
        try:
            _output = [theano.shared(_input)]
        except TypeError as _:
            _output = [_input]
        return T.stack(_output)
    
def concatenate_list(_input):
    if _input is None:
        return None
    elif isinstance(_input, list):
        return T.concatenate(_input, axis=0)
    else:
        return _input
    
    
def closest_to_square_factors(n):
    test = numpy.ceil(numpy.sqrt(float(n)))
    while not (n/test).is_integer():
        test-=1
    if test < 1:
        test = 1
    return int(test), int(n/test)



##################
# PYLEARN2 UTILS #
##################
def get_variable_name(variable, anon="anonymous_variable"):
    """
    If variable has a name, returns that name. Otherwise, returns anon.
    Parameters
    ----------
    variable : tensor_like
        WRITEME
    anon : str, optional
        WRITEME
    Returns
    -------
    WRITEME
    """

    if hasattr(variable, 'name') and variable.name is not None:
        return variable.name

    return anon


def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a shared variable of type floatX
    Parameters
    ----------
    value : WRITEME
    name : WRITEME
    borrow : WRITEME
    dtype : str, optional
        data type. Default value is theano.config.floatX
    Returns
    -------
    WRITEME
    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def as_floatX(variable):
    """
    Casts a given variable into dtype `config.floatX`. Numpy ndarrays will
    remain numpy ndarrays, python floats will become 0-D ndarrays and
    all other types will be treated as theano tensors
    Parameters
    ----------
    variable : WRITEME
    Returns
    -------
    WRITEME
    """

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype
    Parameters
    ----------
    variable : WRITEME
    Returns
    -------
    WRITEME
    """
    return theano.tensor.constant(numpy.asarray(value,
                                             dtype=theano.config.floatX))

def safe_zip(*args):
    """Like zip, but ensures arguments are of same length"""
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)

