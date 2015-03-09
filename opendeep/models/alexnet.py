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
# internal references
from opendeep.models.model import Model
from opendeep.models.conv_layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer

log = logging.getLogger(__name__)

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta,metaopt.verbose=1
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta [metaopt]verbose=1
if theano.config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! To have Theano cherry-pick the best convolution implementation, please set optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(theano.config.optimizer_including))

_defaults = {# data stuff
             "use_data_layer": False,
             "rand_crop": True,
             "shuffle": False,  # whether to shuffle the batches
             "para_load": True,
             "batch_crop_mirror": False,  # if False, do randomly on each image separately
             "batch_size": 256,  # convolutional nets are particular about the batch size
             # conv library
             "lib_conv": 'cudnn'  #lib_conv can be cudnn (recommended) or cudaconvnet
             }

# _train_args = {'batch_size': 256  # convolutional nets are particular about the batch sizes.
#                 }

class AlexNet(Model):
    """
    This is the base model for AlexNet, Alex Krizhevsky's efficient deep convolutional net described in:
    'ImageNet Classification with Deep Convolutional Neural Networks'
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

    Most of the code here is adapted from the authors listed in the license above, from the paper:
    'Theano-based large-scale visual recognition with multiple GPUs'
    Weiguang Ding & Ruoyan Wnag, Fei Mao, Graham Taylor
    http://arxiv.org/pdf/1412.2302.pdf
    """
    def __init__(self, config=None, defaults=_defaults, inputs_hook=None, hiddens_hook=None, dataset=None):
        # init Model to combine the defaults and config dictionaries.
        super(AlexNet, self).__init__(config, defaults)
        # all configuration parameters are now in self.args

        if inputs_hook or hiddens_hook:
            log.critical("Inputs_hook and hiddens_hook not implemented yet for AlexNet!")
            raise NotImplementedError()

        self.dataset = dataset

        self.flag_datalayer = self.args.get('use_data_layer')
        self.lib_conv       = self.args.get('lib_conv')
        self.batch_size     = self.args.get('batch_size')
        self.rand_crop      = self.args.get('rand_crop')

        ####################
        # Theano variables #
        ####################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.rand = T.fvector('rand')

        ##########
        # params #
        ##########
        self.layers = []
        self.params = []
        self.weight_types = []

        self.build_computation_graph()

    def build_computation_graph(self):
        ###################### BUILD NETWORK ##########################
        if self.flag_datalayer:
            data_layer = DataLayer(input=self.x,
                                   image_shape=(3, 256, 256, self.batch_size),
                                   cropsize=227,
                                   rand=self.rand,
                                   mirroring=True,
                                   flag_rand=self.rand_crop)
            layer_1_input = data_layer.get_outputs()
        else:
            layer_1_input = self.x

        convpool_layer1 = ConvPoolLayer(input=layer_1_input,
                                        image_shape=(3, 227, 227, self.batch_size),
                                        filter_shape=(3, 11, 11, 96),
                                        convstride=4,
                                        padsize=0,
                                        group=1,
                                        poolsize=3,
                                        poolstride=2,
                                        bias_init=0.0,
                                        lrn=True,
                                        lib_conv=self.lib_conv)