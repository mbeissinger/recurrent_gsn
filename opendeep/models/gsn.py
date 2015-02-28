'''
@author: Markus Beissinger
University of Pennsylvania, 2014-2015

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

These scripts produce the model trained on MNIST discussed in the paper:
'Deep Generative Stochastic Networks Trainable by Backprop'
Yoshua Bengio, Eric Thibodeau-Laufer
http://arxiv.org/abs/1306.1091

Scheduled noise is added as discussed in the paper:
'Scheduled denoising autoencoders'
Krzysztof J. Geras, Charles Sutton
http://arxiv.org/abs/1406.3269

Multimodal transition operator (using NADE) discussed in:
'Multimodal Transitions for Generative Stochastic Networks'
Sherjil Ozair, Li Yao, Yoshua Bengio
http://arxiv.org/abs/1312.5578
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import os
import cPickle
import time
import logging
# third-party libraries
import numpy
import numpy.random as rng
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
# internal references
import opendeep.log.logger as logger
from opendeep import cast32, function
from opendeep.data.image.mnist import MNIST
from opendeep.models.model import Model
from opendeep.utils.decay_functions import get_decay_function
from opendeep.utils.activation_functions import get_activation_function
from opendeep.utils.cost_functions import get_cost_function
from opendeep.utils import data_tools as data
from opendeep.utils.image_tiler import tile_raster_images
from opendeep.utils.utils import get_shared_weights, get_shared_bias, salt_and_pepper, add_gaussian_noise
from opendeep.utils.utils import make_time_units_string
from opendeep.utils.utils import sharedX, closest_to_square_factors

log = logging.getLogger(__name__)

# Default values to use for some GSN parameters
_defaults = {# gsn parameters
            "layers": 3, # number of hidden layers to use
            "walkbacks": 5, # number of walkbacks (generally 2*layers) - need enough to have info from top layer propagate to visible layer
            "weights_list": None,
            "bias_list": None,
            "hidden_size": 1500,
            "visible_activation": 'sigmoid',
            "hidden_activation": 'tanh',
            "input_sampling": True,
            "MRG": RNG_MRG.MRG_RandomStreams(1),
            # train param
            "cost_function": 'binary_crossentropy',
            # noise parameters
            "noise_annealing": 1.0, #no noise schedule by default
            "add_noise": True,
            "noiseless_h1": True,
            "hidden_add_noise_sigma": 2,
            "input_salt_and_pepper": 0.4,
            # data parameters
            "output_path": '../outputs/gsn/',
            "is_image": True,
            "vis_init": False}

_train_args = {"n_epoch": 1000,
               "batch_size": 128,
               "save_frequency": 10,
               "early_stop_threshold": .9995,
               "early_stop_length": 30,
               "learning_rate": 0.25,
               "lr_decay": "exponential",
               "lr_factor": .995,
               "annealing": 0.995,
               "momentum": 0.5,
               "unsupervised": True}


class GSN(Model):
    '''
    Class for creating a new Generative Stochastic Network (GSN)
    '''
    def __init__(self, config=None, defaults=_defaults, input_hook=None, dataset=None):
        # init Model
        super(self.__class__, self).__init__(config, defaults, input_hook, dataset)
        # now we can access all parameters with self.args! Huzzah!

        self.outdir = self.args.get("output_path")
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'
        data.mkdir_p(self.outdir)
        
        # variables from the dataset that are used for initialization and image reconstruction
        if self.input is None:
            if self.dataset is None:
                self.N_input = self.args.get("input_size")
                if self.args.get("input_size") is None:
                    log.critical("Please either specify input_size in the arguments or provide an example dataset object for input dimensionality.")
                    raise AssertionError("Please either specify input_size in the arguments or provide an example dataset object for input dimensionality.")
            else:
                if len(self.dataset._train_shape) > 1:
                    self.N_input = self.dataset._train_shape[1]
                else:
                    self.N_input = self.dataset._train_shape[0]
        else:
            self.N_input = self.input.shape[1]
        
        self.is_image = self.args.get('is_image')
        if self.is_image:
            (_h, _w) = closest_to_square_factors(self.N_input)
            self.image_width  = self.args.get('width', _w)
            self.image_height = self.args.get('height', _h)
        
        ##########################
        # Network specifications #
        ##########################
        self.layers          = self.args.get('layers')  # number hidden layers
        self.walkbacks       = self.args.get('walkbacks')  # number of walkbacks
        if self.layers % 2 != 0:
            if self.walkbacks < 2*self.layers:
                log.warning('Not enough walkbacks for the layers! Layers is %s and walkbacks is %s. Generaly want 2X walkbacks to layers',str(self.layers), str(self.walkbacks))
        else:
            if self.walkbacks < 2*self.layers-1:
                log.warning('Not enough walkbacks for the layers! Layers is %s and walkbacks is %s. Generaly want 2X walkbacks to layers',str(self.layers), str(self.walkbacks))

        self.noise_annealing = cast32(self.args.get('noise_annealing'))  # exponential noise annealing coefficient
        self.noiseless_h1           = self.args.get('noiseless_h1')
        self.hidden_add_noise_sigma = sharedX(cast32(self.args.get('hidden_add_noise_sigma')))
        self.input_salt_and_pepper  = sharedX(cast32(self.args.get('input_salt_and_pepper')))
        self.input_sampling         = self.args.get('input_sampling')
        self.vis_init               = self.args.get('vis_init')
        
        self.hidden_size = self.args.get('hidden_size')
        self.layer_sizes = [self.N_input] + [self.hidden_size] * self.layers  # layer sizes, from h0 to hK (h0 is the visible layer)
        
        # Activation functions!            
        if callable(self.args.get('hidden_activation')):
            log.debug('Using specified activation for hiddens')
            self.hidden_activation = self.args.get('hidden_activation')
        elif isinstance(self.args.get('hidden_activation'), basestring):
            self.hidden_activation = get_activation_function(self.args.get('hidden_activation'))
            log.debug('Using %s activation for hiddens', self.args.get('hidden_activation'))

        # Visible layer activation
        if callable(self.args.get('visible_activation')):
            log.debug('Using specified activation for visible layer')
            self.visible_activation = self.args.get('visible_activation')
        elif isinstance(self.args.get('visible_activation'), basestring):
            self.visible_activation = get_activation_function(self.args.get('visible_activation'))
            log.debug('Using %s activation for visible layer', self.args.get('visible_activation'))

        # Cost function
        if callable(self.args.get('cost_function')):
            log.debug('Using specified cost function')
            self.cost_function = self.args.get('cost_function')
        elif isinstance(self.args.get('cost_function'), basestring):
            self.cost_function = get_cost_function(self.args.get('cost_function'))
            log.debug('Using %s cost function', self.args.get('cost_function'))


        ############################
        # Theano variables and RNG #
        ############################
        if not self.input:
            self.input = T.fmatrix('X')
        self.MRG = RNG_MRG.MRG_RandomStreams(1)
        rng.seed(1)
        
        ###############
        # Parameters! #
        ###############
        # initialize a list of weights and biases based on layer_sizes for the GSN
        if self.args.get('weights_list') is None:
            self.weights_list = [get_shared_weights(self.layer_sizes[layer], self.layer_sizes[layer+1], name="W_{0!s}_{1!s}".format(layer,layer+1)) for layer in range(self.layers)]  # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
        else:
            self.weights_list = self.config.get('weights_list')

        if self.args.get('bias_list') is None:
            self.bias_list    = [get_shared_bias(self.layer_sizes[layer], name='b_'+str(layer)) for layer in range(self.layers + 1)]  # initialize each layer to 0's.
        else:
            self.bias_list    = self.config.get('bias_list')

        # build the params of the model into a list
        self.params = self.weights_list + self.bias_list
        log.debug("gsn params: %s", str(self.params))

        #################
        # Build the GSN #
        #################
        log.debug("Building GSN graphs")
        # GSN for training - with noise
        add_noise = True
        p_X_chain, _ = GSN.build_gsn(self.input,
                                     self.weights_list,
                                     self.bias_list,
                                     add_noise,
                                     self.noiseless_h1,
                                     self.hidden_add_noise_sigma,
                                     self.input_salt_and_pepper,
                                     self.input_sampling,
                                     self.MRG,
                                     self.visible_activation,
                                     self.hidden_activation,
                                     self.walkbacks)
        
        # GSN for prediction - no noise
        add_noise = False
        p_X_chain_recon, _ = GSN.build_gsn(self.input,
                                           self.weights_list,
                                           self.bias_list,
                                           add_noise,
                                           self.noiseless_h1,
                                           self.hidden_add_noise_sigma,
                                           self.input_salt_and_pepper,
                                           self.input_sampling,
                                           self.MRG,
                                           self.visible_activation,
                                           self.hidden_activation,
                                           self.walkbacks)

        ####################
        # Costs and output #
        ####################
        log.debug('Cost w.r.t p(X|...) at every step in the graph for the GSN')
        # use the noisy ones for training cost
        costs          = [self.cost_function(rX, self.input) for rX in p_X_chain]
        self.show_cost = costs[-1]  # for a monitor to show progress
        self.cost      = numpy.sum(costs)

        # use the non-noisy graph for prediction
        gsn_costs_recon = [self.cost_function(rX, self.input) for rX in p_X_chain_recon]
        self.output     = p_X_chain_recon[-1]
        self.monitor    = gsn_costs_recon[-1]
        

        ############
        # Sampling #
        ############
        # the input to the sampling function
        X_sample = T.fmatrix("X_sampling")
        self.network_state_input = [X_sample] + [T.fmatrix("H_sampling_"+str(i+1)) for i in range(self.layers)]
       
        # "Output" state of the network (noisy)
        # initialized with input, then we apply updates
        self.network_state_output = [X_sample] + self.network_state_input[1:]
        visible_pX_chain = []
    
        # ONE update
        log.debug("Performing one walkback in network state sampling.")
        GSN.update_layers(self.network_state_output,
                          self.weights_list,
                          self.bias_list,
                          visible_pX_chain,
                          True,
                          self.noiseless_h1,
                          self.hidden_add_noise_sigma,
                          self.input_salt_and_pepper,
                          self.input_sampling,
                          self.MRG,
                          self.visible_activation,
                          self.hidden_activation)

        #################################
        #     Create the functions      #
        #################################
        log.debug("Compiling functions...")
        t = time.time()

        log.debug("f_predict...")
        self.f_predict = function(inputs  = [self.input],
                                  outputs = self.output,
                                  name    = 'gsn_f_predict')
        

        log.debug("f_noise...")
        self.f_noise = function(inputs  = [self.input],
                                outputs = salt_and_pepper(self.input, self.input_salt_and_pepper, self.MRG),
                                name    = 'gsn_f_noise')
    
        log.debug("f_sample...")
        if self.layers == 1: 
            self.f_sample = function(inputs  = [X_sample],
                                     outputs = visible_pX_chain[-1],
                                     name    = 'gsn_f_sample_single_layer')
        else:
            # WHY IS THERE A WARNING????
            # because the first odd layers are not used -> directly computed FROM THE EVEN layers
            # unused input = warn
            self.f_sample = function(inputs  = self.network_state_input,
                                     outputs = self.network_state_output + visible_pX_chain,
                                     name    = 'gsn_f_sample')


        # # things for log likelihood
        # self.H = T.tensor3('H', dtype='float32')
        # add_noise = True
        # if add_noise:
        #     x_init = salt_and_pepper(self.input, self.input_salt_and_pepper, self.MRG)
        # else:
        #     x_init = self.input
        # hiddens = [x_init]+[self.H[i] for i in range(len(self.bias_list)-1)]
        # sample = GSN.build_gsn_pxh(hiddens, self.weights_list, self.bias_list, add_noise, self.noiseless_h1, self.hidden_add_noise_sigma, self.input_salt_and_pepper, self.input_sampling, self.MRG, self.visible_activation, self.hidden_activation, self.walkbacks, self.logger)
        #
        # log.debug("P(X=x|H)")
        # self.pxh = function(inputs = [self.input, self.H], outputs=sample, name='px_given_h ')
        #

        log.debug("GSN compiling done. Took %s", make_time_units_string(time.time() - t))
        
        
    def train(self, dataset=None, optimizer=None, optimizer_config=_train_args, rng=rng):
        '''
        The method to train this model, given a dataset object (and an optimizer).
        Use generic Stochastic Gradient Descent by default (although look into using ADADELTA by default because it is
        easier without learning rate parameter)

        :param dataset: opendeep.data.dataset
        The dataset object to train on.
        :param optimizer: opendeep.optimization.optimizer
        The optimizer object to train with.
        :param optimizer_config: dictionary-like or json/yaml file
        The configuration for the optimizer
        :param rng: rng-like object
        The random number generator (same interface as numpy's random)
        :return: None
        '''
        super(self.__class__, self).train(dataset, optimizer, optimizer_config, rng)

    def get_train_cost(self):
        '''
        This returns the expression that represents the cost given an input, which is used for the optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the optimizer object.

        :return: expression (theano tensor)
        an expression giving the cost of the model for training.
        '''
        return self.cost

    def get_train_params(self):
        '''
        This returns a list of the model parameters to be trained by the optimizer.

        :return: list
        Model parameters to be trained.
        '''
        return self.params

    def get_decay_params(self):
        '''
        This returns a list of the opendeep.utils.decay_functions on internal parameters to decay each time step.
        Used with the optimizer, for example, to decay things like GSN noise over time (noise scheduling).
        '''
        # noise scheduling
        noise_schedule = get_decay_function('exponential', self.input_salt_and_pepper, self.args.get('input_salt_and_pepper'), self.noise_annealing)
        return [noise_schedule]

    def get_monitors(self):
        '''
        :return: list
        A list of the theano variables to use as the monitors during training - the model variables you want to check
        periodically during training.
        '''
        return [self.cost, self.show_cost, self.monitor]

    
    
    def gen_10k_samples(self):
        log.info('Generating 10,000 samples')
        samples, _ = self.sample(self.test_X[0].get_value()[1:2], 10000, 1)
        f_samples = 'samples.npy'
        numpy.save(f_samples, samples)
        log.debug('saved digits')
        
    def sample(self, initial, n_samples=400, k=1):
        log.debug("Starting sampling...")
        def sample_some_numbers_single_layer(n_samples):
            x0 = initial
            samples = [x0]
            x = self.f_noise(x0)
            for _ in xrange(n_samples-1):
                x = self.f_sample(x)
                samples.append(x)
                x = rng.binomial(n=1, p=x, size=x.shape).astype('float32')
                x = self.f_noise(x)
                
            log.debug("Sampling done.")
            return numpy.vstack(samples), None
        
        def sampling_wrapper(NSI):
            # * is the "splat" operator: It takes a list as input, and expands it into actual positional arguments in the function call.
            out = self.f_sample(*NSI)
            NSO = out[:len(self.network_state_output)]
            vis_pX_chain = out[len(self.network_state_output):]
            return NSO, vis_pX_chain
        
        def sample_some_numbers(n_samples):
            # The network's initial state
            init_vis       = initial
            noisy_init_vis = self.f_noise(init_vis)
            
            network_state  = [[noisy_init_vis] + [numpy.zeros((initial.shape[0],self.hidden_size), dtype='float32') for _ in self.bias_list[1:]]]
            
            visible_chain  = [init_vis]
            noisy_h0_chain = [noisy_init_vis]
            sampled_h = []
            
            times = []
            for i in xrange(n_samples-1):
                _t = time.time()
               
                # feed the last state into the network, compute new state, and obtain visible units expectation chain 
                net_state_out, vis_pX_chain = sampling_wrapper(network_state[-1])
    
                # append to the visible chain
                visible_chain += vis_pX_chain
    
                # append state output to the network state chain
                network_state.append(net_state_out)
                
                noisy_h0_chain.append(net_state_out[0])
                
                if i%k == 0:
                    sampled_h.append(T.stack(net_state_out[1:]))
                    if i == k:
                        log.debug("About "+make_time_units_string(numpy.mean(times)*(n_samples-1-i))+" remaining...")
                    
                times.append(time.time() - _t)
    
            log.DEBUG("Sampling done.")
            return numpy.vstack(visible_chain), sampled_h
        
        if self.layers == 1:
            return sample_some_numbers_single_layer(n_samples)
        else:
            return sample_some_numbers(n_samples)
        
    def plot_samples(self, epoch_number="", leading_text="", n_samples=400):
        to_sample = time.time()
        initial = self.test_X[0].get_value(borrow=True)[:1]
        rand_idx = numpy.random.choice(range(self.test_X[0].get_value(borrow=True).shape[0]))
        rand_init = self.test_X[0].get_value(borrow=True)[rand_idx:rand_idx+1]
        
        V, _ = self.sample(initial, n_samples)
        rand_V, _ = self.sample(rand_init, n_samples)
        
        img_samples = PIL.Image.fromarray(tile_raster_images(V, (self.image_height, self.image_width), closest_to_square_factors(n_samples)))
        rand_img_samples = PIL.Image.fromarray(tile_raster_images(rand_V, (self.image_height, self.image_width), closest_to_square_factors(n_samples)))
        
        fname = self.outdir+leading_text+'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname)
        rfname = self.outdir+leading_text+'samples_rand_epoch_'+str(epoch_number)+'.png'
        rand_img_samples.save(rfname) 
        log.debug('Took ' + make_time_units_string(time.time() - to_sample) + ' to sample '+str(n_samples*2)+' numbers')
        
    #############################
    # Save the model parameters #
    #############################
    def save_params(self, n='', params=[]):
        '''
        This saves the model paramaters to the param_file (pickle file)

        :param param_file: filename of pickled params file
        :return: Boolean
        whether successful
        '''
        log.info('saving parameters...')
        save_path = self.outdir+'gsn_params'+n+'.pkl'
        f = open(save_path, 'wb')
        try:
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close()
            
    def load_params(self, filename):
        '''
        This loads and sets the model paramaters found in the param_file (pickle file)

        :param param_file: filename of pickled params file
        :return: Boolean
        whether successful
        '''
        if os.path.isfile(filename):
            log.info("Loading existing GSN parameters...")
            loaded_params = cPickle.load(open(filename,'r'))
            [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[:len(self.weights_list)], self.weights_list)]
            [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[len(self.weights_list):], self.bias_list)]
            log.info("Parameters loaded.")
        else:
            log.info("Could not find existing GSN parameter file %s.", filename)
        



###############################################
# COMPUTATIONAL GRAPH HELPER METHODS FOR GSN #
###############################################
    @staticmethod
    def update_layers(hiddens,
                      weights_list,
                      bias_list,
                      p_X_chain,
                      add_noise              = _defaults["add_noise"],
                      noiseless_h1           = _defaults["noiseless_h1"],
                      hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                      input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                      input_sampling         = _defaults["input_sampling"],
                      MRG                    = _defaults["MRG"],
                      visible_activation     = _defaults["visible_activation"],
                      hidden_activation      = _defaults["hidden_activation"],
                      logger = None):
        # One update over the odd layers + one update over the even layers
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('done full update.')

    @staticmethod
    def update_layers_scan_step(hiddens_t,
                                weights_list,
                                bias_list,
                                add_noise              = _defaults["add_noise"],
                                noiseless_h1           = _defaults["noiseless_h1"],
                                hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                                input_sampling         = _defaults["input_sampling"],
                                MRG                    = _defaults["MRG"],
                                visible_activation     = _defaults["visible_activation"],
                                hidden_activation      = _defaults["hidden_activation"],
                                logger = None):
        p_X_chain = []
        log.debug("One full update step for layers.")
        # One update over the odd layers + one update over the even layers
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens_t, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens_t, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('done full update.')
        # return the generated sample, the sampled next input, and hiddens
        return p_X_chain[0], hiddens_t

    @staticmethod
    def update_layers_reverse(hiddens,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise              = _defaults["add_noise"],
                              noiseless_h1           = _defaults["noiseless_h1"],
                              hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                              input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                              input_sampling         = _defaults["input_sampling"],
                              MRG                    = _defaults["MRG"],
                              visible_activation     = _defaults["visible_activation"],
                              hidden_activation      = _defaults["hidden_activation"],
                              logger = None):
        # One update over the even layers + one update over the odd layers
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        log.debug('done full update.')


    # Odd layer update function
    # just a loop over the odd layers
    @staticmethod
    def update_odd_layers(hiddens,
                          weights_list,
                          bias_list,
                          add_noise              = _defaults["add_noise"],
                          noiseless_h1           = _defaults["noiseless_h1"],
                          hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                          input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                          input_sampling         = _defaults["input_sampling"],
                          MRG                    = _defaults["MRG"],
                          visible_activation     = _defaults["visible_activation"],
                          hidden_activation      = _defaults["hidden_activation"],
                          logger = None):
        # Loop over the odd layers
        for i in range(1, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            GSN.simple_update_layer(hiddens, weights_list, bias_list, None, i, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

    # Even layer update
    # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
    @staticmethod
    def update_even_layers(hiddens,
                           weights_list,
                           bias_list,
                           p_X_chain,
                           add_noise              = _defaults["add_noise"],
                           noiseless_h1           = _defaults["noiseless_h1"],
                           hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                           input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                           input_sampling         = _defaults["input_sampling"],
                           MRG                    = _defaults["MRG"],
                           visible_activation     = _defaults["visible_activation"],
                           hidden_activation      = _defaults["hidden_activation"],
                           logger = None):
        # Loop over even layers
        for i in range(0, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            GSN.simple_update_layer(hiddens, weights_list, bias_list, p_X_chain, i, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)


    # The layer update function
    # hiddens   :   list containing the symbolic theano variables [visible, hidden1, hidden2, ...]
    #               layer_update will modify this list inplace
    # weights_list : list containing the theano variables weights between hidden layers
    # bias_list :   list containing the theano variables bias corresponding to hidden layers
    # p_X_chain :   list containing the successive p(X|...) at each update
    #               update_layer will append to this list
    # i         :   the current layer being updated
    # add_noise :   pre (and post) activation gaussian noise flag
    # logger    :   specified Logger to use for output messages
    @staticmethod
    def simple_update_layer(hiddens,
                            weights_list,
                            bias_list,
                            p_X_chain,
                            i,
                            add_noise              = _defaults["add_noise"],
                            noiseless_h1           = _defaults["noiseless_h1"],
                            hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                            input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                            input_sampling         = _defaults["input_sampling"],
                            MRG                    = _defaults["MRG"],
                            visible_activation     = _defaults["visible_activation"],
                            hidden_activation      = _defaults["hidden_activation"],
                            logger = None):
        # Compute the dot product, whatever layer
        # If the visible layer X
        if i == 0:
            log.debug('using '+str(weights_list[i])+'.T')
            hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]
        # If the top layer
        elif i == len(hiddens)-1:
            log.debug('using '+str(weights_list[i-1]))
            hiddens[i] = T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
        # Otherwise in-between layers
        else:
            log.debug("using %s and %s.T", str(weights_list[i-1]), str(weights_list[i]))
            # next layer        :   hiddens[i+1], assigned weights : W_i
            # previous layer    :   hiddens[i-1], assigned weights : W_(i-1)
            hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i == 1 and noiseless_h1:
            log.debug('>>NO noise in first hidden layer')
            add_noise = False

        # pre activation noise
        if i != 0 and add_noise:
            log.debug('Adding pre-activation gaussian noise for layer %s', str(i))
            hiddens[i] = add_gaussian_noise(hiddens[i], hidden_add_noise_sigma, MRG)

        # ACTIVATION!
        if i == 0:
            log.debug('Activation for visible layer')
            hiddens[i] = visible_activation(hiddens[i])
        else:
            log.debug('Hidden units activation for layer %s', str(i))
            hiddens[i] = hidden_activation(hiddens[i])

        # post activation noise
        # why is there post activation noise? Because there is already pre-activation noise, this just doubles the amount of noise between each activation of the hiddens.
        if i != 0 and add_noise:
            log.debug('Adding post-activation gaussian noise for layer %s', str(i))
            hiddens[i] = add_gaussian_noise(hiddens[i], hidden_add_noise_sigma, MRG)

        # build the reconstruction chain if updating the visible layer X
        if i == 0:
            # if input layer -> append p(X|H...)
            p_X_chain.append(hiddens[i])

            # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
            if input_sampling:
                log.debug('Sampling from input')
                sampled = MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
                log.debug('>>NO input sampling')
                sampled = hiddens[i]
            # add noise
            sampled = salt_and_pepper(sampled, input_salt_and_pepper, MRG)

            # set input layer
            hiddens[i] = sampled



    ############################
    #   THE MAIN GSN BUILDER   #
    ############################
    @staticmethod
    def build_gsn(X,
                  weights_list,
                  bias_list,
                  add_noise              = _defaults["add_noise"],
                  noiseless_h1           = _defaults["noiseless_h1"],
                  hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                  input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                  input_sampling         = _defaults["input_sampling"],
                  MRG                    = _defaults["MRG"],
                  visible_activation     = _defaults["visible_activation"],
                  hidden_activation      = _defaults["hidden_activation"],
                  walkbacks              = _defaults["walkbacks"],
                  logger = None):
        """
        Construct a GSN (unimodal transition operator) for k walkbacks on the input X.
        Returns the list of predicted X's after k walkbacks and the resulting layer values.

        @type  X: Theano symbolic variable
        @param X: The variable representing the visible input.

        @type  weights_list: List(matrix)
        @param weights_list: The list of weights to use between layers.

        @type  bias_list: List(vector)
        @param bias_list: The list of biases to use for each layer.

        @type  add_noise: Boolean
        @param add_noise: Whether or not to add noise in the computational graph.

        @type  noiseless_h1: Boolean
        @param noiseless_h1: Whether or not to add noise in the first hidden layer.

        @type  hidden_add_noise_sigma: Float
        @param hidden_add_noise_sigma: The sigma value for the hidden noise function.

        @type  input_salt_and_pepper: Float
        @param input_salt_and_pepper: The amount of masking noise to use.

        @type  input_sampling: Boolean
        @param input_sampling: Whether to sample from each walkback prediction (like Gibbs).

        @type  MRG: Theano random generator
        @param MRG: Random generator.

        @type  visible_activation: Function
        @param visible_activation: The visible layer X activation function.

        @type  hidden_activation: Function
        @param hidden_activation: The hidden layer activation function.

        @type  walkbacks: Integer
        @param walkbacks: The k number of walkbacks to use for the GSN.

        @type  logger: Logger
        @param logger: The output log to use.

        @rtype:   List
        @return:  predicted_x_chain, hiddens
        """
        p_X_chain = []
        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(X, input_salt_and_pepper, MRG)
        else:
            X_init = X
        # init hiddens with zeros
        hiddens = [X_init]
        for w in weights_list:
            hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
        # The layer update scheme
        log.info("Building the GSN graph : %s updates", str(walkbacks))
        for i in range(walkbacks):
            log.debug("GSN Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

        return p_X_chain, hiddens

    @staticmethod
    def build_gsn_given_hiddens(X,
                                hiddens,
                                weights_list,
                                bias_list,
                                add_noise              = _defaults["add_noise"],
                                noiseless_h1           = _defaults["noiseless_h1"],
                                hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                                input_sampling         = _defaults["input_sampling"],
                                MRG                    = _defaults["MRG"],
                                visible_activation     = _defaults["visible_activation"],
                                hidden_activation      = _defaults["hidden_activation"],
                                walkbacks              = _defaults["walkbacks"],
                                cost_function          = _defaults["cost_function"],
                                logger = None):

        log.info("Building the GSN graph given hiddens with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN (prediction) Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers_reverse(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

        x_sample = p_X_chain[-1]

        costs     = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1] # for logging to show progress
        cost      = numpy.sum(costs)

        return x_sample, cost, show_cost

    @staticmethod
    def build_gsn_scan(X,
                       weights_list,
                       bias_list,
                       add_noise              = _defaults["add_noise"],
                       noiseless_h1           = _defaults["noiseless_h1"],
                       hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                       input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                       input_sampling         = _defaults["input_sampling"],
                       MRG                    = _defaults["MRG"],
                       visible_activation     = _defaults["visible_activation"],
                       hidden_activation      = _defaults["hidden_activation"],
                       walkbacks              = _defaults["walkbacks"],
                       cost_function          = _defaults["cost_function"],
                       logger = None):

        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(X, input_salt_and_pepper, MRG)
        else:
            X_init = X
        # init hiddens with zeros
        hiddens_0 = [X_init]
        for w in weights_list:
            hiddens_0.append(T.zeros_like(T.dot(hiddens_0[-1], w)))

        log.info("Building the GSN graph (for scan) with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN (after scan) Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens_0, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)


        x_sample = p_X_chain[-1]

        costs     = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1] # for logging to show progress
        cost      = numpy.sum(costs)

        return x_sample, cost, show_cost#, updates

    @staticmethod
    def build_gsn_pxh(hiddens,
                    weights_list,
                    bias_list,
                    add_noise              = _defaults["add_noise"],
                    noiseless_h1           = _defaults["noiseless_h1"],
                    hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                    input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                    input_sampling         = _defaults["input_sampling"],
                    MRG                    = _defaults["MRG"],
                    visible_activation     = _defaults["visible_activation"],
                    hidden_activation      = _defaults["hidden_activation"],
                    walkbacks              = _defaults["walkbacks"],
                    logger = None):

        log.info("Building the GSN graph for P(X=x|H) with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

        x_sample = p_X_chain[-1]

        return x_sample





###############################################
# MAIN METHOD FOR RUNNING DEFAULT GSN EXAMPLE #
###############################################
def main():
    ########################################
    # Initialization things with arguments #
    ########################################
    logger.config_root_logger()
    log.info("Creating a new GSN")

    data = MNIST()
    gsn = GSN(dataset=data)

    # # Load initial weights and biases from file if testing
    # params_to_load = 'gsn_params.pkl'
    # if test_model:
    #     gsn.load_params(params_to_load)

    gsn.train()



if __name__ == '__main__':
    main()