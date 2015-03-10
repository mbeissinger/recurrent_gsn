"""
.. module:: nnet

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

and theano_alexnet (https://github.com/uoguelph-mlrg/theano_alexnet)
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import numpy
import theano
import theano.tensor as T
# internal imports
from opendeep import cast_floatX

log = logging.getLogger(__name__)

numpy.random.RandomState(23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

# these are the possible formulas for the interval when building weights from a uniform distribution
_uniform_interval = {
    'sigmoid': lambda n_row, n_col: 4 * numpy.sqrt(6. / (n_row + n_col)),  # use this only when the activation function is sigmoid
    'default': lambda n_row, n_col: 1 / numpy.sqrt(n_row),  # this is the default provided in other codebases
    'good'   : lambda n_row, n_col: numpy.sqrt(6. / (n_row + n_col))  # this is the default for the GSN code from Li Yao
}
def get_weights_uniform(shape, interval=None, name="W", rng=None):
    """
    This initializes a shared variable with a given shape for weights drawn from a Uniform distribution with
    low = -interval and high = interval.

    Interval can either be a number to use, or a string key to one of the predefined formulas in the _uniform_interval dictionary.

    :param shape: a tuple giving the shape information for this weight matrix
    :type shape: Tuple

    :param interval: either a number for your own custom interval, or a string key to one of the predefined formulas
    :type interval: Float or String

    :param name: the name to give the shared variable
    :type name: String

    :param rng: the random number generator to use with a .uniform method
    :type rng: random

    :return: the theano shared variable with given shape and name drawn from a uniform distribution
    :rtype: shared variable

    :raises: NotImplementedError
    """
    default_interval = 'good'

    interval = interval or default_interval

    if rng is None:
        rng = numpy.random
    # If the interval parameter is a string, grab the appropriate formula from the function dictionary, and apply the appropriate
    # shape numbers to it.
    if isinstance(interval, basestring):
        interval_func = _uniform_interval.get(interval)
        if interval_func is None:
            log.error('Could not find uniform interval formula %s, try one of %s instead.' %
                      str(interval), str(_uniform_interval.keys()))
            raise NotImplementedError('Could not find uniform interval formula %s, try one of %s instead.' %
                                      str(interval), str(_uniform_interval.keys())
            )
        else:
            log.debug("Creating weights with shape %s from Uniform distribution with formula name: %s", str(shape), str(interval))
            # if this is a 2D weight matrix (normally the case), the tuple is formatted (n_rows, n_columns)
            if len(shape) == 2:
                interval = interval_func(shape[0], shape[1])
            else:
                log.error("Expected the shape to be 2D - was %s. If you are calling this from a convolutional layer (4D), please specify interval on your own. This is to make it easier to deal with bc01 vs c01b formats.", str(len(shape)))
                raise NotImplementedError(
                    "Expected the shape to be 2D - was %s. If you are calling this from a convolutional layer (4D), please specify interval on your own. This is to make it easier to deal with bc01 vs c01b formats."%
                    str(len(shape))
                )
    else:
        log.debug("Creating weights with shape %s from Uniform distribution with given interval +- %s", str(shape), str(interval))
    # build the uniform weights tensor
    val = cast_floatX(rng.uniform(low=-interval, high=interval, size=shape))
    return theano.shared(value=val, name=name)

def get_weights_gaussian(shape, mean=None, std=None, name="W", rng=None):
    """
    This initializes a shared variable with the given shape for weights drawn from a Gaussian distribution with mean and std.

    :param shape: a tuple giving the shape information for this weight matrix
    :type shape: Tuple

    :param mean: the mean to use for the Gaussian distribution
    :type mean: float

    :param std: the standard deviation to use dor the Gaussian distribution
    :type std: float

    :param name: the name to give the shared variable
    :type name: String

    :param rng: a given random number generator to use with .normal method
    :type rng: random

    :return: the theano shared variable with given shape and drawn from a Gaussian distribution
    :rtype: shared variable
    """
    default_mean = 0
    default_std  = 0.05

    mean = mean or default_mean
    std = std or default_std

    log.debug("Creating weights with shape %s from Gaussian mean=%s, std=%s", str(shape), str(mean), str(std))
    if rng is None:
        rng = numpy.random

    if std != 0:
        val = numpy.asarray(rng.normal(loc=mean, scale=std, size=shape), dtype=theano.config.floatX)
    else:
        val = cast_floatX(mean * numpy.ones(shape, dtype=theano.config.floatX))

    return theano.shared(value=val, name=name)

def get_bias(shape, name="b", init_values=None):
    """
    This creates a theano shared variable for the bias parameter - normally initialized to zeros, but you can specify other values

    :param shape: the shape to use for the bias vector/matrix
    :type shape: Tuple

    :param name: the name to give the shared variable
    :type name: String

    :param offset: values to add to the zeros, if you want a nonzero bias initially
    :type offset: float/vector

    :return: the theano shared variable with given shape
    :rtype: shared variable
    """
    default_init = 0

    init_values = init_values or default_init

    log.debug("Initializing bias variable with shape %s" % str(shape))
    # init to zeros plus the offset
    val = cast_floatX(numpy.ones(shape=shape, dtype=theano.config.floatX) * init_values)
    return theano.shared(value=val, name=name)

def mirror_images(input, image_shape, cropsize, rand, flag_rand):
    """
    This takes an input batch of images (normally the input to a convolutional net), and augments them by mirroring and concatenating.

    :param input: the input 4D tensor of images
    :type input: Tensor4D

    :param image_shape: the shape of the 4D tensor input
    :type image_shape: Tuple

    :param cropsize: what size to crop to
    :type cropsize: Integer

    :param rand: a vector representing a random array for cropping/mirroring the data
    :type rand: fvector

    :param flag_rand: to randomize the mirror
    :type flag_rand: Boolean

    :return: tensor4D representing the mirrored/concatenated input
    :rtype: same as input
    """
    # The random mirroring and cropping in this function is done for the
    # whole batch.

    # trick for random mirroring
    mirror = input[:, :, ::-1, :]
    input = T.concatenate([input, mirror], axis=0)

    # crop images
    center_margin = (image_shape[2] - cropsize) / 2

    if flag_rand:
        mirror_rand = T.cast(rand[2], 'int32')
        crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
        crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
    else:
        mirror_rand = 0
        crop_xs = center_margin
        crop_ys = center_margin

    output = input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
    output = output[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

    log.debug("mirrored input data with shape_in: " + str(image_shape))

    return output