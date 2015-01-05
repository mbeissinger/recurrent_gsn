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

import os, cPickle, time, argparse
import numpy, theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import numpy.random as rng
from numpy import ceil, sqrt
import PIL.Image
from collections import OrderedDict
from utils import logger as log
from utils import data_tools as data
from utils.image_tiler import tile_raster_images
from utils.utils import cast32, logit, trunc, get_shared_weights, get_shared_bias, salt_and_pepper, add_gaussian_noise, make_time_units_string, load_from_config

# Default values to use for some GSN parameters
defaults = {# gsn parameters
            "layers": 3, # number of hidden layers to use
            "walkbacks": 5, # number of walkbacks (generally 2*layers) - need enough to have info from top layer propagate to visible layer
            "hidden_size": 1500,
            "visible_activation": lambda x: T.nnet.sigmoid(x),
            "hidden_activation": lambda x: T.tanh(x),
            "input_sampling": True,
            "MRG": RNG_MRG.MRG_RandomStreams(1),
            # training parameters
            "cost_function": lambda x,y: T.mean(T.nnet.binary_crossentropy(x,y)),
            "n_epoch": 1000,
            "batch_size": 100,
            "save_frequency": 10,
            "early_stop_threshold": .9995,
            "early_stop_length": 30,
            "learning_rate": 0.25,
            "annealing": 0.995,
            "momentum": 0.5,
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


class GSN:
    '''
    Class for creating a new Generative Stochastic Network (GSN)
    '''
    def __init__(self, train_X=None, valid_X=None, test_X=None, args=None, logger=None):
        # Output logger
        self.logger = logger
        self.outdir = args.get("output_path", defaults["output_path"])
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'
        # Input data
        self.train_X = train_X
        self.valid_X = valid_X
        self.test_X  = test_X
        
        # variables from the dataset that are used for initialization and image reconstruction
        if train_X is None:
            self.N_input = args.get("input_size")
            if args.get("input_size") is None:
                raise AssertionError("Please either specify input_size in the arguments or provide an example train_X for input dimensionality.")
        else:
            self.N_input = train_X.eval().shape[1]
        self.root_N_input = numpy.sqrt(self.N_input)
        
        self.is_image = args.get('is_image', defaults['is_image'])
        if self.is_image:
            self.image_width  = args.get('width', self.root_N_input)
            self.image_height = args.get('height', self.root_N_input)
        
        #######################################
        # Network and training specifications #
        #######################################
        self.layers          = args.get('layers', defaults['layers']) # number hidden layers
        self.walkbacks       = args.get('walkbacks', defaults['walkbacks']) # number of walkbacks
        self.learning_rate   = theano.shared(cast32(args.get('learning_rate', defaults['learning_rate'])))  # learning rate
        self.init_learn_rate = cast32(args.get('learning_rate', defaults['learning_rate']))
        self.momentum        = theano.shared(cast32(args.get('momentum', defaults['momentum']))) # momentum term
        self.annealing       = cast32(args.get('annealing', defaults['annealing'])) # exponential annealing coefficient
        self.noise_annealing = cast32(args.get('noise_annealing', defaults['noise_annealing'])) # exponential noise annealing coefficient
        self.batch_size      = args.get('batch_size', defaults['batch_size'])
        self.n_epoch         = args.get('n_epoch', defaults['n_epoch'])
        self.early_stop_threshold = args.get('early_stop_threshold', defaults['early_stop_threshold'])
        self.early_stop_length = args.get('early_stop_length', defaults['early_stop_length'])
        self.save_frequency  = args.get('save_frequency', defaults['save_frequency'])
        
        self.noiseless_h1           = args.get('noiseless_h1', defaults["noiseless_h1"])
        self.hidden_add_noise_sigma = theano.shared(cast32(args.get('hidden_add_noise_sigma', defaults["hidden_add_noise_sigma"])))
        self.input_salt_and_pepper  = theano.shared(cast32(args.get('input_salt_and_pepper', defaults["input_salt_and_pepper"])))
        self.input_sampling         = args.get('input_sampling', defaults["input_sampling"])
        self.vis_init               = args.get('vis_init', defaults['vis_init'])
        
        self.layer_sizes = [self.N_input] + [args.get('hidden_size', defaults['hidden_size'])] * self.layers # layer sizes, from h0 to hK (h0 is the visible layer)
        
        self.f_recon = None
        self.f_noise = None
        
        # Activation functions!
        if args.get('hidden_activation') is not None:
            log.maybeLog(self.logger, 'Using specified activation for hiddens')
            self.hidden_activation = args.get('hidden_activation')
        elif args.get('hidden_act') == 'sigmoid':
            log.maybeLog(self.logger, 'Using sigmoid activation for hiddens')
            self.hidden_activation = T.nnet.sigmoid
        elif args.get('hidden_act') == 'rectifier':
            log.maybeLog(self.logger, 'Using rectifier activation for hiddens')
            self.hidden_activation = lambda x : T.maximum(cast32(0), x)
        elif args.get('hidden_act') == 'tanh':
            log.maybeLog(self.logger, 'Using hyperbolic tangent activation for hiddens')
            self.hidden_activation = lambda x : T.tanh(x)
        elif args.get('hidden_act') is not None:
            log.maybeLog(self.logger, "Did not recognize hidden activation {0!s}, please use tanh, rectifier, or sigmoid".format(args.get('hidden_act')))
            raise NotImplementedError("Did not recognize hidden activation {0!s}, please use tanh, rectifier, or sigmoid".format(args.get('hidden_act')))
        else:
            log.maybeLog(self.logger, "Using default activation for hiddens")
            self.hidden_activation = defaults['hidden_activation']
        # Visible layer activation
        if args.get('visible_activation') is not None:
            log.maybeLog(self.logger, 'Using specified activation for visible layer')
            self.visible_activation = args.get('visible_activation')
        elif args.get('visible_act') == 'sigmoid':
            log.maybeLog(self.logger, 'Using sigmoid activation for visible layer')
            self.visible_activation = T.nnet.sigmoid
        elif args.get('visible_act') == 'softmax':
            log.maybeLog(self.logger, 'Using softmax activation for visible layer')
            self.visible_activation = T.nnet.softmax
        elif args.get('visible_act') is not None:
            log.maybeLog(self.logger, "Did not recognize visible activation {0!s}, please use sigmoid or softmax".format(args.get('visible_act')))
            raise NotImplementedError("Did not recognize visible activation {0!s}, please use sigmoid or softmax".format(args.get('visible_act')))
        else:
            log.maybeLog(self.logger, 'Using default activation for visible layer')
            self.visible_activation = defaults['visible_activation']
            
        # Cost function!
        if args.get('cost_function') is not None:
            log.maybeLog(self.logger, '\nUsing specified cost function for GSN training\n')
            self.cost_function = args.get('cost_function')
        elif args.get('cost_funct') == 'binary_crossentropy':
            log.maybeLog(self.logger, '\nUsing binary cross-entropy cost!\n')
            self.cost_function = lambda x,y: T.mean(T.nnet.binary_crossentropy(x,y))
        elif args.get('cost_funct') == 'square':
            log.maybeLog(self.logger, "\nUsing square error cost!\n")
            #cost_function = lambda x,y: T.log(T.mean(T.sqr(x-y)))
            self.cost_function = lambda x,y: T.log(T.sum(T.pow((x-y),2)))
        elif args.get('cost_funct') is not None:
            log.maybeLog(self.logger, "\nDid not recognize cost function {0!s}, please use binary_crossentropy or square\n".format(args.get('cost_funct')))
            raise NotImplementedError("Did not recognize cost function {0!s}, please use binary_crossentropy or square".format(args.get('cost_funct')))
        else:
            log.maybeLog(self.logger, '\nUsing default cost function for GSN training\n')
            self.cost_function = defaults['cost_function']
        
        ############################
        # Theano variables and RNG #
        ############################
        self.X   = T.fmatrix('X') # for use in sampling
        self.MRG = RNG_MRG.MRG_RandomStreams(1)
        rng.seed(1)
        
        ###############
        # Parameters! #
        ###############
        # initialize a list of weights and biases based on layer_sizes for the GSN
        if args.get('weights_list') is None:
            self.weights_list = [get_shared_weights(self.layer_sizes[layer], self.layer_sizes[layer+1], name="W_{0!s}_{1!s}".format(layer,layer+1)) for layer in range(self.layers)] # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
        else:
            self.weights_list = args.get('weights_list')
        if args.get('bias_list') is None:
            self.bias_list    = [get_shared_bias(self.layer_sizes[layer], name='b_'+str(layer)) for layer in range(self.layers + 1)] # initialize each layer to 0's.
        else:
            self.bias_list    = args.get('bias_list')
        self.params = self.weights_list + self.bias_list
        
        #################
        # Build the GSN #
        #################
        log.maybeLog(self.logger, "\nBuilding GSN graphs for training and testing")
        # GSN for training - with noise
        add_noise = True
        p_X_chain, _ = build_gsn(self.X,
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
                                 self.walkbacks,
                                 self.logger)
        
        # GSN for reconstruction checks along the way - no noise
        add_noise = False
        p_X_chain_recon, _ = build_gsn(self.X,
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
                                       self.walkbacks,
                                       self.logger)
        
        #######################
        # Costs and gradients #
        #######################
        log.maybeLog(self.logger, 'Cost w.r.t p(X|...) at every step in the graph for the GSN')
        gsn_costs     = [self.cost_function(rX, self.X) for rX in p_X_chain]
        show_gsn_cost = gsn_costs[-1] # for logging to show progress
        gsn_cost      = numpy.sum(gsn_costs)
        
        gsn_costs_recon     = [self.cost_function(rX, self.X) for rX in p_X_chain_recon]
        show_gsn_cost_recon = gsn_costs_recon[-1]
        
        log.maybeLog(self.logger, ["gsn params:", self.params])
        
        # Stochastic gradient descent!
        gradient        =   T.grad(gsn_cost, self.params)              
        gradient_buffer =   [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in self.params] 
        m_gradient      =   [self.momentum * gb + (cast32(1) - self.momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
        param_updates   =   [(param, param - self.learning_rate * mg) for (param, mg) in zip(self.params, m_gradient)]
        gradient_buffer_updates = zip(gradient_buffer, m_gradient)
        updates         =   OrderedDict(param_updates + gradient_buffer_updates)
        
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
        log.maybeLog(self.logger, "Performing one walkback in network state sampling.")
        update_layers(self.network_state_output,
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
                      self.hidden_activation,
                      self.logger)
        
        #################################
        #     Create the functions      #
        #################################
        log.maybeLog(self.logger, "Compiling functions...")
        t = time.time()
        
        self.f_learn = theano.function(inputs  = [self.X],
                                  updates = updates,
                                  outputs = show_gsn_cost,
                                  name='gsn_f_learn')
    
        self.f_cost  = theano.function(inputs  = [self.X],
                                  outputs = show_gsn_cost,
                                  name='gsn_f_cost')
        
        # used for checkpoints and testing - no noise in network
        self.f_recon = theano.function(inputs  = [self.X],
                                       outputs = [show_gsn_cost_recon, p_X_chain_recon[-1]],
                                       name='gsn_f_recon')
        
        self.f_noise = theano.function(inputs = [self.X],
                                       outputs = salt_and_pepper(self.X, self.input_salt_and_pepper),
                                       name='gsn_f_noise')
    
        if self.layers == 1: 
            self.f_sample = theano.function(inputs = [X_sample], 
                                            outputs = visible_pX_chain[-1], 
                                            name='gsn_f_sample_single_layer')
        else:
            # WHY IS THERE A WARNING????
            # because the first odd layers are not used -> directly computed FROM THE EVEN layers
            # unused input = warn
            self.f_sample = theano.function(inputs = self.network_state_input,
                                            outputs = self.network_state_output + visible_pX_chain,
                                            on_unused_input='warn',
                                            name='gsn_f_sample')
        
        log.maybeLog(self.logger, "Compiling done. Took "+make_time_units_string(time.time() - t)+".\n")
        
        
    def train(self, train_X=None, valid_X=None, test_X=None, continue_training=False):
        log.maybeLog(self.logger, "\nTraining---------\n")
        if train_X is None:
            log.maybeLog(self.logger, "Training using data given during initialization of GSN.\n")
            train_X = self.train_X
            if train_X is None:
                log.maybeLog(self.logger, "\nPlease provide a training dataset!\n")
                raise AssertionError("Please provide a training dataset!")
        else:
            log.maybeLog(self.logger, "Training using data provided to training function.\n")
        if valid_X is None:
            valid_X = self.valid_X
        if test_X is None:
            test_X  = self.test_X
            
        
        ############
        # TRAINING #
        ############
        log.maybeLog(self.logger, "-----------TRAINING GSN FOR {0!s} EPOCHS-----------".format(self.n_epoch))
        STOP        = False
        counter     = 0
        if not continue_training:
            self.learning_rate.set_value(self.init_learn_rate)  # learning rate
        times       = []
        best_cost   = float('inf')
        best_params = None
        patience    = 0
                    
        log.maybeLog(self.logger, ['train X size:',str(train_X.shape.eval())])
        if valid_X is not None:
            log.maybeLog(self.logger, ['valid X size:',str(valid_X.shape.eval())])
        if test_X is not None:
            log.maybeLog(self.logger, ['test X size:',str(test_X.shape.eval())])
        
        if self.vis_init:
            self.bias_list[0].set_value(logit(numpy.clip(0.9,0.001,train_X.get_value().mean(axis=0))))
    
        while not STOP:
            counter += 1
            t = time.time()
            log.maybeAppend(self.logger, [counter,'\t'])
            
            #shuffle the data
            data.shuffle_data(train_X)
            data.shuffle_data(valid_X)
            data.shuffle_data(test_X)
            
            #train
            train_costs = data.apply_cost_function_to_dataset(self.f_learn, train_X, self.batch_size)
            log.maybeAppend(self.logger, ['Train:',trunc(train_costs), '\t'])
    
            #valid
            if valid_X is not None:
                valid_costs = data.apply_cost_function_to_dataset(self.f_cost, valid_X, self.batch_size)
                log.maybeAppend(self.logger, ['Valid:',trunc(valid_costs), '\t'])
    
            #test
            if test_X is not None:
                test_costs = data.apply_cost_function_to_dataset(self.f_cost, test_X, self.batch_size)
                log.maybeAppend(self.logger, ['Test:',trunc(test_costs), '\t'])
                
            #check for early stopping
            if valid_X is not None:
                cost = numpy.sum(valid_costs)
            else:
                cost = numpy.sum(train_costs)
            if cost < best_cost*self.early_stop_threshold:
                patience = 0
                best_cost = cost
                # save the parameters that made it the best
                best_params = save_params(self.params)
            else:
                patience += 1
    
            if counter >= self.n_epoch or patience >= self.early_stop_length:
                STOP = True
                if best_params is not None:
                    restore_params(self.params, best_params)
                save_params_to_file(counter, self.params, self.outdir, self.logger)
    
            timing = time.time() - t
            times.append(timing)
    
            log.maybeAppend(self.logger, 'time: '+make_time_units_string(timing)+'\t')
            
            log.maybeLog(self.logger, 'remaining: '+make_time_units_string((self.n_epoch - counter) * numpy.mean(times)))
        
            if (counter % self.save_frequency) == 0 or STOP is True:
                if self.is_image:
                    n_examples = 100
                    tests = test_X.get_value()[0:n_examples]
                    noisy_tests = self.f_noise(test_X.get_value()[0:n_examples])
                    _, reconstructed = self.f_recon(noisy_tests) 
                    # Concatenate stuff if it is an image
                    stacked = numpy.vstack([numpy.vstack([tests[i*10 : (i+1)*10], noisy_tests[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
                    number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (self.image_height,self.image_width), (10,30)))
                    
                    number_reconstruction.save(self.outdir+'gsn_image_reconstruction_epoch_'+str(counter)+'.png')
        
                #save gsn_params
                save_params_to_file(counter, self.params, self.outdir, self.logger)
         
            # ANNEAL!
            new_lr = self.learning_rate.get_value() * self.annealing
            self.learning_rate.set_value(new_lr)
            
            new_hidden_sigma = self.hidden_add_noise_sigma.get_value() * self.noise_annealing
            self.hidden_add_noise_sigma.set_value(new_hidden_sigma)
            
            new_salt_pepper = self.input_salt_and_pepper.get_value() * self.noise_annealing
            self.input_salt_and_pepper.set_value(new_salt_pepper)
            
        
    
    
    
    def test(self, test_X=None):
        log.maybeLog(self.logger, "\nTesting---------\n")
        if test_X is None:
            log.maybeLog(self.logger, "Testing using data given during initialization of GSN.\n")
            test_X  = self.test_X
            if test_X is None:
                log.maybeLog(self.logger, "\nPlease provide a test dataset!\n")
                raise AssertionError("Please provide a test dataset")
        else:
            log.maybeLog(self.logger, "Testing using data provided to test function.\n")
            
        ###########
        # TESTING #
        ###########
        n_examples = 100
        tests = test_X.get_value()[0:n_examples]
        noisy_tests = self.f_noise(test_X.get_value()[0:n_examples])
        cost, reconstructed = self.f_recon(noisy_tests) 
        # Concatenate stuff if it is an image
        if self.is_image:
            stacked = numpy.vstack([numpy.vstack([tests[i*10 : (i+1)*10], noisy_tests[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
            number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (self.image_height,self.image_width), (10,30)))
            
            number_reconstruction.save(self.outdir+'gsn_image_reconstruction_test.png')
        # Otherwise, save reconstructed numpy array as csv
        else:
            numpy.savetxt(self.outdir+'gsn_reconstruction_test.csv', reconstructed, delimiter=",")
            
        log.maybeLog(self.logger, "----------------\n\nAverage test cost is "+str(cost)+"\n\n-----------------")
        
    
    
    
    def sample(self, initial, n_samples=400):
        def sample_some_numbers_single_layer(n_samples):
            x0 = initial
            samples = [x0]
            x = self.f_noise(x0)
            for _ in xrange(n_samples-1):
                x = self.f_sample(x)
                samples.append(x)
                x = rng.binomial(n=1, p=x, size=x.shape).astype('float32')
                x = self.f_noise(x)
                
            return numpy.vstack(samples)
                
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
    
            network_state  = [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in self.bias_list[1:]]]
    
            visible_chain  = [init_vis]
    
            noisy_h0_chain = [noisy_init_vis]
    
            for _ in xrange(n_samples-1):
               
                # feed the last state into the network, compute new state, and obtain visible units expectation chain 
                net_state_out, vis_pX_chain = sampling_wrapper(network_state[-1])
    
                # append to the visible chain
                visible_chain += vis_pX_chain
    
                # append state output to the network state chain
                network_state.append(net_state_out)
                
                noisy_h0_chain.append(net_state_out[0])
    
            return numpy.vstack(visible_chain)#, numpy.vstack(noisy_h0_chain)
        
        if self.layers == 1:
            return sample_some_numbers_single_layer(n_samples)
        else:
            return sample_some_numbers(n_samples)
        
    def plot_samples(self, epoch_number="", leading_text="", n_samples=400):
        to_sample = time.time()
        initial = self.test_X.get_value()[:1]
        V = self.sample(initial, n_samples)
        img_samples = PIL.Image.fromarray(tile_raster_images(V, (self.root_N_input,self.root_N_input), (ceil(sqrt(n_samples)), ceil(sqrt(n_samples)))))
        
        fname = self.outdir+leading_text+'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        log.maybeLog(self.logger, 'Took ' + str(time.time() - to_sample) + ' to sample '+n_samples+' numbers')
            
        
         
    
    


###############################################
# COMPUTATIONAL GRAPH HELPER METHODS FOR GSN #
###############################################
def update_layers(hiddens,
                  weights_list,
                  bias_list,
                  p_X_chain, 
                  add_noise              = defaults["add_noise"],
                  noiseless_h1           = defaults["noiseless_h1"],
                  hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                  input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                  input_sampling         = defaults["input_sampling"],
                  MRG                    = defaults["MRG"],
                  visible_activation     = defaults["visible_activation"],
                  hidden_activation      = defaults["hidden_activation"],
                  logger = None):
    # One update over the odd layers + one update over the even layers
    log.maybeLog(logger, 'odd layer updates')
    # update the odd layers
    update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'even layer updates')
    # update the even layers
    update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'done full update.\n')
    
def update_layers_scan_step(hiddens_t,
                            weights_list,
                            bias_list,
                            add_noise              = defaults["add_noise"],
                            noiseless_h1           = defaults["noiseless_h1"],
                            hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                            input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                            input_sampling         = defaults["input_sampling"],
                            MRG                    = defaults["MRG"],
                            visible_activation     = defaults["visible_activation"],
                            hidden_activation      = defaults["hidden_activation"],
                            logger = None):
    p_X_chain = []
    log.maybeLog(logger, "One full update step for layers.")
    # One update over the odd layers + one update over the even layers
    log.maybeLog(logger, 'odd layer updates')
    # update the odd layers
    update_odd_layers(hiddens_t, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'even layer updates')
    # update the even layers
    update_even_layers(hiddens_t, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'done full update.\n')
    # return the generated sample, the sampled next input, and hiddens
    return p_X_chain[0], hiddens_t
        
    
def update_layers_reverse(hiddens,
                          weights_list,
                          bias_list,
                          p_X_chain, 
                          add_noise              = defaults["add_noise"],
                          noiseless_h1           = defaults["noiseless_h1"],
                          hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                          input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                          input_sampling         = defaults["input_sampling"],
                          MRG                    = defaults["MRG"],
                          visible_activation     = defaults["visible_activation"],
                          hidden_activation      = defaults["hidden_activation"],
                          logger = None):
    # One update over the even layers + one update over the odd layers
    log.maybeLog(logger, 'even layer updates')
    # update the even layers
    update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'odd layer updates')
    # update the odd layers
    update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
    log.maybeLog(logger, 'done full update.\n')
        
        
# Odd layer update function
# just a loop over the odd layers
def update_odd_layers(hiddens,
                      weights_list,
                      bias_list,
                      add_noise              = defaults["add_noise"],
                      noiseless_h1           = defaults["noiseless_h1"],
                      hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                      input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                      input_sampling         = defaults["input_sampling"],
                      MRG                    = defaults["MRG"],
                      visible_activation     = defaults["visible_activation"],
                      hidden_activation      = defaults["hidden_activation"],
                      logger = None):
    # Loop over the odd layers
    for i in range(1, len(hiddens), 2):
        log.maybeLog(logger, ['updating layer',i])
        simple_update_layer(hiddens, weights_list, bias_list, None, i, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

# Even layer update
# p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
def update_even_layers(hiddens,
                       weights_list,
                       bias_list,
                       p_X_chain,
                       add_noise              = defaults["add_noise"],
                       noiseless_h1           = defaults["noiseless_h1"],
                       hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                       input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                       input_sampling         = defaults["input_sampling"],
                       MRG                    = defaults["MRG"],
                       visible_activation     = defaults["visible_activation"],
                       hidden_activation      = defaults["hidden_activation"],
                       logger = None):
    # Loop over even layers
    for i in range(0, len(hiddens), 2):
        log.maybeLog(logger, ['updating layer',i])
        simple_update_layer(hiddens, weights_list, bias_list, p_X_chain, i, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        

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
def simple_update_layer(hiddens,
                        weights_list,
                        bias_list,
                        p_X_chain,
                        i,
                        add_noise              = defaults["add_noise"],
                        noiseless_h1           = defaults["noiseless_h1"],
                        hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                        input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                        input_sampling         = defaults["input_sampling"],
                        MRG                    = defaults["MRG"],
                        visible_activation     = defaults["visible_activation"],
                        hidden_activation      = defaults["hidden_activation"],
                        logger = None):   
    # Compute the dot product, whatever layer
    # If the visible layer X
    if i == 0:
        log.maybeLog(logger, 'using '+str(weights_list[i])+'.T')
        hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]           
    # If the top layer
    elif i == len(hiddens)-1:
        log.maybeLog(logger, ['using',weights_list[i-1]])
        hiddens[i] = T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
    # Otherwise in-between layers
    else:
        log.maybeLog(logger, ["using {0!s} and {1!s}.T".format(weights_list[i-1], weights_list[i])])
        # next layer        :   hiddens[i+1], assigned weights : W_i
        # previous layer    :   hiddens[i-1], assigned weights : W_(i-1)
        hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

    # Add pre-activation noise if NOT input layer
    if i==1 and noiseless_h1:
        log.maybeLog(logger, '>>NO noise in first hidden layer')
        add_noise = False

    # pre activation noise       
    if i != 0 and add_noise:
        log.maybeLog(logger, ['Adding pre-activation gaussian noise for layer', i])
        hiddens[i] = add_gaussian_noise(hiddens[i], hidden_add_noise_sigma)
   
    # ACTIVATION!
    if i == 0:
        log.maybeLog(logger, 'Activation for visible layer')
        hiddens[i] = visible_activation(hiddens[i])
    else:
        log.maybeLog(logger, ['Hidden units activation for layer', i])
        hiddens[i] = hidden_activation(hiddens[i])

    # post activation noise
    # why is there post activation noise? Because there is already pre-activation noise, this just doubles the amount of noise between each activation of the hiddens.  
    if i != 0 and add_noise:
        log.maybeLog(logger, ['Adding post-activation gaussian noise for layer', i])
        hiddens[i] = add_gaussian_noise(hiddens[i], hidden_add_noise_sigma)

    # build the reconstruction chain if updating the visible layer X
    if i == 0:
        # if input layer -> append p(X|...)
        p_X_chain.append(hiddens[i])
        
        # sample from p(X|...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
        if input_sampling:
            log.maybeLog(logger, 'Sampling from input')
            sampled = MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
        else:
            log.maybeLog(logger, '>>NO input sampling')
            sampled = hiddens[i]
        # add noise
        sampled = salt_and_pepper(sampled, input_salt_and_pepper)
        
        # set input layer
        hiddens[i] = sampled



############################
#   THE MAIN GSN BUILDER   #
############################
def build_gsn(X,
              weights_list,
              bias_list,
              add_noise              = defaults["add_noise"],
              noiseless_h1           = defaults["noiseless_h1"],
              hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
              input_salt_and_pepper  = defaults["input_salt_and_pepper"],
              input_sampling         = defaults["input_sampling"],
              MRG                    = defaults["MRG"],
              visible_activation     = defaults["visible_activation"],
              hidden_activation      = defaults["hidden_activation"],
              walkbacks              = defaults["walkbacks"],
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
        X_init = salt_and_pepper(X, input_salt_and_pepper)
    else:
        X_init = X
    # init hiddens with zeros
    hiddens = [X_init]
    for w in weights_list:
        hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
    # The layer update scheme
    log.maybeLog(logger, ["Building the GSN graph :", walkbacks,"updates"])
    for i in range(walkbacks):
        log.maybeLog(logger, "GSN Walkback {!s}/{!s}".format(i+1,walkbacks))
        update_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        
    return p_X_chain, hiddens


def build_gsn_given_hiddens(X,
                            hiddens,
                            weights_list,
                            bias_list,
                            add_noise              = defaults["add_noise"],
                            noiseless_h1           = defaults["noiseless_h1"],
                            hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                            input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                            input_sampling         = defaults["input_sampling"],
                            MRG                    = defaults["MRG"],
                            visible_activation     = defaults["visible_activation"],
                            hidden_activation      = defaults["hidden_activation"],
                            walkbacks              = defaults["walkbacks"],
                            cost_function          = defaults["cost_function"],
                            logger = None):
    
    log.maybeLog(logger, ["Building the GSN graph given hiddens with", walkbacks,"walkbacks"])
    p_X_chain = []
    for i in range(walkbacks):
        log.maybeLog(logger, "GSN (prediction) Walkback {!s}/{!s}".format(i+1,walkbacks))
        update_layers_reverse(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        

    x_sample = p_X_chain[-1]
    
    costs     = [cost_function(rX, X) for rX in p_X_chain]
    show_cost = costs[-1] # for logging to show progress
    cost      = numpy.sum(costs)
    
    return x_sample, cost, show_cost


def build_gsn_scan(X,
                   weights_list,
                   bias_list,
                   add_noise              = defaults["add_noise"],
                   noiseless_h1           = defaults["noiseless_h1"],
                   hidden_add_noise_sigma = defaults["hidden_add_noise_sigma"],
                   input_salt_and_pepper  = defaults["input_salt_and_pepper"],
                   input_sampling         = defaults["input_sampling"],
                   MRG                    = defaults["MRG"],
                   visible_activation     = defaults["visible_activation"],
                   hidden_activation      = defaults["hidden_activation"],
                   walkbacks              = defaults["walkbacks"],
                   cost_function          = defaults["cost_function"],
                   logger = None):
    
    # Whether or not to corrupt the visible input X
    if add_noise:
        X_init = salt_and_pepper(X, input_salt_and_pepper)
    else:
        X_init = X
    # init hiddens with zeros
    hiddens_0 = [X_init]
    for w in weights_list:
        hiddens_0.append(T.zeros_like(T.dot(hiddens_0[-1], w)))
    
    log.maybeLog(logger, ["Building the GSN graph with", walkbacks,"walkbacks"])
    p_X_chain = []
    for i in range(walkbacks):
        log.maybeLog(logger, "GSN (after scan) Walkback {!s}/{!s}".format(i+1,walkbacks))
        update_layers(hiddens_0, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1, hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        

    x_sample = p_X_chain[-1]
    
    costs     = [cost_function(rX, X) for rX in p_X_chain]
    show_cost = costs[-1] # for logging to show progress
    cost      = numpy.sum(costs)
    
    return x_sample, cost, show_cost#, updates
    



#############################
# Save the model parameters #
#############################
def save_params_to_file(epoch, gsn_params, outdir, logger=None):
    logger.log('saving parameters...')
    if outdir[-1] != '/':
        outdir = outdir + '/'
    save_path = outdir+'gsn_params_epoch_'+str(epoch)+'.pkl'
    f = open(save_path, 'wb')
    try:
        cPickle.dump(gsn_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()
        
def save_params(params):
    values = [param.get_value(borrow=True) for param in params]
    return values

def restore_params(params, values):
    for i in range(len(params)):
        params[i].set_value(values[i])
        
        
        

###############################################
# MAIN METHOD FOR RUNNING DEFAULT GSN EXAMPLE #
###############################################
def main():
    parser = argparse.ArgumentParser()

    # GSN settings
    parser.add_argument('--layers', type=int, default=3) # number of hidden layers
    parser.add_argument('--walkbacks', type=int, default=5) # number of walkbacks
    parser.add_argument('--hidden_size', type=int, default=1500)
    parser.add_argument('--hidden_act', type=str, default='tanh')
    parser.add_argument('--visible_act', type=str, default='sigmoid')
    
    # training
    parser.add_argument('--cost_funct', type=str, default='binary_crossentropy') # the cost function for training
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_frequency', type=int, default=10) #number of epochs between parameters being saved
    parser.add_argument('--early_stop_threshold', type=float, default=0.9995)
    parser.add_argument('--early_stop_length', type=int, default=30) #the patience number of epochs
    
    # noise
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2)
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4) #default=0.4
    
    # hyper parameters
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--noise_annealing', type=float, default=0.99)
    
    # data
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--output_path', type=str, default='../outputs/gsn/')
   
    # argparse does not deal with booleans
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1)
    parser.add_argument('--input_sampling', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--continue_training', type=int, default=0) #default=0
    
    args = parser.parse_args()
    
    ########################################
    # Initialization things with arguments #
    ########################################
    outdir = args.output_path + "/" + args.dataset + "/"
    data.mkdir_p(outdir)
    args.output_path = outdir
    
    # Create the logger
    logger = log.Logger(outdir)
    logger.log("---------CREATING GSN------------\n\n")
    logger.log(args)
    
    # See if we should load args from a previous config file (during testing)
    config_filename = outdir+'config'
    if args.test_model and 'config' in os.listdir(outdir):
        config_vals = load_from_config(config_filename)
        for CV in config_vals:
            logger.log(CV)
            if CV.startswith('test'):
                logger.log('Do not override testing switch')
                continue        
            try:
                exec('args.'+CV) in globals(), locals()
            except:
                exec('args.'+CV.split('=')[0]+"='"+CV.split('=')[1]+"'") in globals(), locals()
    else:
        # Save the current configuration
        # Useful for logs/experiments
        logger.log('Saving config')
        with open(config_filename, 'w') as f:
            f.write(str(args))
            
    ######################################
    # Load the data, train = train+valid #
    ######################################
    if args.dataset.lower() == 'mnist':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist(args.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))
    else:
        raise AssertionError("Dataset not recognized. Please try MNIST, or implement your own data processing method in data_tools.py")

    # transfer the datasets into theano shared variables
    train_X, train_Y = data.shared_dataset((train_X, train_Y), borrow=True)
    valid_X, valid_Y = data.shared_dataset((valid_X, valid_Y), borrow=True)
    test_X, test_Y   = data.shared_dataset((test_X, test_Y), borrow=True)
     
    ##########################        
    # Initialize the new GSN #
    ##########################
    gsn = GSN(train_X, valid_X, test_X, vars(args), logger)
    
    # Load initial weights and biases from file if testing
    params_to_load = 'gsn_params.pkl'
    if args.test_model and os.path.isfile(params_to_load):
        logger.log("\nLoading existing GSN parameters")
        loaded_params = cPickle.load(open(params_to_load,'r'))
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[:len(gsn.weights_list)], gsn.weights_list)]
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[len(gsn.weights_list):], gsn.bias_list)]
    else:
        logger.log("Could not find existing GSN parameter file {}, training instead.".format(params_to_load))
        args.test_model = False
    
    #########################################
    # Train or test the new GSN on the data #
    #########################################
    # Train if not test
    if not args.test_model:
        gsn.train()
    # Otherwise, test
    else:
        gsn.test()


if __name__ == '__main__':
    main()