"""
Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Weiguang Ding", "Ruoyan Wang", "Fei Mao", "Graham Taylor", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.expr.normalize import CrossChannelNormalization
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
# internal references
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights_gaussian, get_bias
from opendeep.utils.activation import rectifier


log = logging.getLogger(__name__)

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta,metaopt.verbose=1
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta [metaopt]verbose=1
if theano.config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! To have Theano cherry-pick the best convolution implementation, please set optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(theano.config.optimizer_including))

class ConvPoolLayer(Model):
    """
    A fast convolutional and pooling layer combo for AlexNet implementation of authors above.
    """
    def __init__(self, inputs_hook, filter_shape, convstride, padsize, group, poolsize, poolstride, bias_init,
                 local_response_normalization=False, lib_conv='cudnn', params_hook=None, config=None, defaults=None):
        # init Model to combine the defaults and config dictionaries.
        super(ConvPoolLayer, self).__init__(config, defaults)
        # all configuration parameters are now in self.args

        # deal with the inputs coming from inputs_hook
        self.image_shape = inputs_hook[0]
        self.input = inputs_hook[1]

        # layer configuration TODO: move this to config files?
        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = self.image_shape[0]
        self.lrn = local_response_normalization
        self.lib_conv = lib_conv
        self.group = group
        assert self.group in [1, 2]

        self.filter_shape = numpy.asarray(filter_shape)
        self.image_shape = numpy.asarray(self.image_shape)

        if self.lrn:
            self.lrn_func = CrossChannelNormalization()

        # Params - make sure to deal with params_hook!
        if self.group == 1:
            if params_hook:
                assert len(params_hook) == 2
                self.W, self.b = params_hook
            else:
                self.W = get_weights_gaussian(shape=self.filter_shape, mean=0, std=0.01, name="W")
                self.b = get_bias(shape=self.filter_shape[3], init_values=bias_init, name="b")
            self.params = [self.W, self.b]
        else:
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[3] = self.filter_shape[3] / 2

            self.image_shape[0] = self.image_shape[0] / 2
            self.image_shape[3] = self.image_shape[3] / 2
            if params_hook:
                assert len(params_hook) == 4
                self.W0, self.W1, self.b0, self.b1 = params_hook
            else:
                self.W0 = get_weights_gaussian(shape=self.filter_shape, name="W0")
                self.W1 = get_weights_gaussian(shape=self.filter_shape, name="W1")
                self.b0 = get_bias(shape=self.filter_shape[3], init_values=bias_init, name="b0")
                self.b1 = get_bias(shape=self.filter_shape[3], init_values=bias_init, name="b1")
            self.params = [self.W0, self.b0, self.W1, self.b1]

        if lib_conv == 'cudaconvnet':
            self.build_cudaconvnet_graph()
        elif lib_conv == 'cudnn':
            self.build_cudnn_graph()
        else:
            log.error("The lib_conv %s is not supported! Please choose cudaconvnet or cudnn"%str(lib_conv))
            raise NotImplementedError("The lib_conv %s is not supported! Please choose cudaconvnet or cudnn"%str(lib_conv))

        # Local Response Normalization (for AlexNet)
        if self.lrn:
            # lrn_input = gpu_contiguous(self.output)
            self.output = self.lrn_func(self.output)

        log.debug("conv (%s) layer with shape_in: %s" % lib_conv, str(self.image_shape))

    def build_cudaconvnet_graph(self):
            self.conv_op = FilterActs(pad=self.padsize, stride=self.convstride, partial_sum=1)

            # Conv
            if self.group == 1:
                contiguous_input = gpu_contiguous(input)
                contiguous_filters = gpu_contiguous(self.W)

                conv_out = self.conv_op(contiguous_input, contiguous_filters)
                conv_out = conv_out + self.b.dimshuffle(0, 'x', 'x', 'x')
            else:
                contiguous_input0 = gpu_contiguous(input[:self.channel / 2, :, :, :])
                contiguous_filters0 = gpu_contiguous(self.W0)

                conv_out0 = self.conv_op(contiguous_input0, contiguous_filters0)
                conv_out0 = conv_out0 + self.b0.dimshuffle(0, 'x', 'x', 'x')

                contiguous_input1 = gpu_contiguous(input[self.channel / 2:, :, :, :])
                contiguous_filters1 = gpu_contiguous(self.W1)

                conv_out1 = self.conv_op(contiguous_input1, contiguous_filters1)
                conv_out1 = conv_out1 + self.b1.dimshuffle(0, 'x', 'x', 'x')

                conv_out = T.concatenate([conv_out0, conv_out1], axis=0)

            # ReLu
            self.output = rectifier(conv_out)
            conv_out = gpu_contiguous(conv_out)

            # Pooling!
            if self.poolsize != 1:
                self.pool_op = MaxPool(ds=self.poolsize, stride=self.poolstride)

            self.output = self.pool_op(self.output)

    def build_cudnn_graph(self):
        input_shuffled = self.input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        # in01out to outin01
        # print image_shape_shuffled
        # print filter_shape_shuffled

        if self.group == 1:
            W_shuffled = self.W.dimshuffle(3, 0, 1, 2)  # c01b to bc01

            conv_out = dnn.dnn_conv(img=input_shuffled,
                                    kerns=W_shuffled,
                                    subsample=(self.convstride, self.convstride),
                                    border_mode=self.padsize)
            conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        else:
            W0_shuffled = self.W0.dimshuffle(3, 0, 1, 2)  # c01b to bc01

            conv_out0 = dnn.dnn_conv(img=input_shuffled[:, :self.channel / 2, :, :],
                                     kerns=W0_shuffled,
                                     subsample=(self.convstride, self.convstride),
                                     border_mode=self.padsize)
            conv_out0 = conv_out0 + self.b0.dimshuffle('x', 0, 'x', 'x')

            W1_shuffled = self.W1.dimshuffle(3, 0, 1, 2)  # c01b to bc01

            conv_out1 = dnn.dnn_conv(img=input_shuffled[:, self.channel / 2:, :, :],
                                     kerns=W1_shuffled,
                                     subsample=(self.convstride, self.convstride),
                                     border_mode=self.padsize)
            conv_out1 = conv_out1 + self.b1.dimshuffle('x', 0, 'x', 'x')

            conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

        # ReLu
        self.output = rectifier(conv_out)

        # Pooling
        if self.poolsize != 1:
            self.output = dnn.dnn_pool(self.output,
                                       ws=(self.poolsize, self.poolsize),
                                       stride=(self.poolstride, self.poolstride))

        self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

    def get_inputs(self):
        """
        This should return the input(s) to the model's computation graph. This is called by the Optimizer when creating
        the theano train function on the cost expression returned by get_train_cost().

        This should normally return the same theano variable list that is used in the inputs= argument to the f_predict
        function.
        ------------------

        :return: Theano variables representing the input(s) to the training function.
        :rtype: List(theano variable)
        """
        return [self.input]

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph. This should be what is given for the
        outputs= part of the 'f_predict' function from self.predict().

        This will be used for creating hooks to link models together, where these outputs can be strung as the inputs or hiddens to another
        model :)
        ------------------

        :return: theano expression of the outputs from this model's computation
        :rtype: theano tensor (expression)
        """
        return self.output

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the Optimizer. These parameters are used in the gradient.
        ------------------

        :return: flattened list of theano shared variables to be trained
        :rtype: List(shared_variables)
        """
        return self.params