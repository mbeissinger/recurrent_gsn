'''
@author: Markus Beissinger
University of Pennsylvania, 2014-2015

This class produces the model discussed in the paper: (my rnn-gsn paper)

Inspired by code for the RNN-RBM:
http://deeplearning.net/tutorial/rnnrbm.html

'''

import numpy, os, sys, cPickle
import numpy.random as rng
import random as R
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
import time
from utils import data_tools as data
import warnings
from utils.logger import Logger
from hessian_free.hf import hf_optimizer as hf_optimizer
from hessian_free.hf import SequenceDataset as hf_sequence_dataset
import gsn
import utils.logger as log
from utils.image_tiler import tile_raster_images
from utils.utils import cast32, logit, trunc, get_shared_weights, get_shared_bias, salt_and_pepper, make_time_units_string, get_activation_function, get_cost_function, raise_to_list, closest_to_square_factors, copy_params, restore_params
from numpy import ceil, sqrt


# Default values to use for some RNN-GSN parameters
defaults = {# gsn parameters
            "layers": 3, # number of hidden layers to use
            "walkbacks": 5, # number of walkbacks (generally 2*layers) - need enough to have info from top layer propagate to visible layer
            "hidden_size": 1500,
            "hidden_activation": lambda x: T.tanh(x),
            "visible_activation": lambda x: T.nnet.sigmoid(x),
            "input_sampling": True,
            "MRG": RNG_MRG.MRG_RandomStreams(1),
            # recurrent parameters
            "recurrent_hidden_size": 1500,
            "recurrent_hidden_activation": lambda x: T.tanh(x),
            # training parameters
            "initialize_gsn": True,
            "cost_function": lambda x,y: T.mean(T.nnet.binary_crossentropy(x,y)),
            "n_epoch": 1000,
            "gsn_batch_size": 100,
            "batch_size": 200,
            "save_frequency": 10,
            "early_stop_threshold": .9995,
            "early_stop_length": 30,
            "hessian_free": False,
            "learning_rate": 0.25,
            "annealing": 0.995,
            "momentum": 0.5,
            "regularize_weight": 0,
            # noise parameters
            "add_noise": True,
            "noiseless_h1": True,
            "hidden_add_noise_sigma": 2,
            "input_salt_and_pepper": 0.4,
            "noise_annealing": 1.0, #no noise schedule by default
            # data parameters
            "is_image": True,
            "vis_init": False,
            "output_path": '../outputs/rnn_gsn/'}


class RNN_GSN():
    '''
    Class for creating a new Recurrent Generative Stochastic Network (RNN-GSN)
    '''
    def __init__(self, train_X=None, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None, args=None, logger=None):
        # Output logger
        self.logger = logger
        self.outdir = args.get("output_path", defaults["output_path"])
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'
            
        data.mkdir_p(self.outdir)
 
        # Input data - make sure it is a list of shared datasets if it isn't. THIS WILL KEEP 'NONE' AS 'NONE' no need to worry :)
        self.train_X = raise_to_list(train_X)
        self.train_Y = raise_to_list(train_Y)
        self.valid_X = raise_to_list(valid_X)
        self.valid_Y = raise_to_list(valid_Y)
        self.test_X  = raise_to_list(test_X)
        self.test_Y  = raise_to_list(test_Y)
        
        # variables from the dataset that are used for initialization and image reconstruction
        if train_X is None:
            self.N_input = args.get("input_size")
            if args.get("input_size") is None:
                raise AssertionError("Please either specify input_size in the arguments or provide an example train_X for input dimensionality.")
        else:
            self.N_input = train_X[0].eval().shape[1]
        
        self.is_image = args.get('is_image', defaults['is_image'])
        if self.is_image:
            (_h, _w) = closest_to_square_factors(self.N_input)
            self.image_width  = args.get('width', _w)
            self.image_height = args.get('height', _h)
            
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
        self.gsn_batch_size = args.get('gsn_batch_size', defaults['gsn_batch_size'])
        self.n_epoch         = args.get('n_epoch', defaults['n_epoch'])
        self.early_stop_threshold = args.get('early_stop_threshold', defaults['early_stop_threshold'])
        self.early_stop_length = args.get('early_stop_length', defaults['early_stop_length'])
        self.save_frequency  = args.get('save_frequency', defaults['save_frequency'])
        
        self.noiseless_h1           = args.get('noiseless_h1', defaults["noiseless_h1"])
        self.hidden_add_noise_sigma = theano.shared(cast32(args.get('hidden_add_noise_sigma', defaults["hidden_add_noise_sigma"])))
        self.input_salt_and_pepper  = theano.shared(cast32(args.get('input_salt_and_pepper', defaults["input_salt_and_pepper"])))
        self.input_sampling         = args.get('input_sampling', defaults["input_sampling"])
        self.vis_init               = args.get('vis_init', defaults['vis_init'])
        self.initialize_gsn         = args.get('initialize_gsn', defaults['initialize_gsn'])
        self.hessian_free           = args.get('hessian_free', defaults['hessian_free'])
        
        self.hidden_size = args.get('hidden_size', defaults['hidden_size'])
        self.layer_sizes = [self.N_input] + [self.hidden_size] * self.layers # layer sizes, from h0 to hK (h0 is the visible layer)
        self.recurrent_hidden_size = args.get('recurrent_hidden_size', defaults['recurrent_hidden_size'])
        
        self.f_recon = None
        self.f_noise = None
        
        # Activation functions!
        # For the GSN:
        if args.get('hidden_activation') is not None:
            log.maybeLog(self.logger, 'Using specified activation for GSN hiddens')
            self.hidden_activation = args.get('hidden_activation')
        elif args.get('hidden_act') is not None:
            self.hidden_activation = get_activation_function(args.get('hidden_act'))
            log.maybeLog(self.logger, 'Using {0!s} activation for GSN hiddens'.format(args.get('hidden_act')))
        else:
            log.maybeLog(self.logger, "Using default activation for GSN hiddens")
            self.hidden_activation = defaults['hidden_activation']
            
        # For the RNN:
        if args.get('recurrent_hidden_activation') is not None:
            log.maybeLog(self.logger, 'Using specified activation for RNN hiddens')
            self.recurrent_hidden_activation = args.get('recurrent_hidden_activation')
        elif args.get('recurrent_hidden_act') is not None:
            self.recurrent_hidden_activation = get_activation_function(args.get('recurrent_hidden_act'))
            log.maybeLog(self.logger, 'Using {0!s} activation for RNN hiddens'.format(args.get('recurrent_hidden_act')))
        else:
            log.maybeLog(self.logger, "Using default activation for RNN hiddens")
            self.recurrent_hidden_activation = defaults['recurrent_hidden_activation']
            
        # Visible layer activation
        if args.get('visible_activation') is not None:
            log.maybeLog(self.logger, 'Using specified activation for visible layer')
            self.visible_activation = args.get('visible_activation')
        elif args.get('visible_act') is not None:
            self.visible_activation = get_activation_function(args.get('visible_act'))
            log.maybeLog(self.logger, 'Using {0!s} activation for visible layer'.format(args.get('visible_act')))
        else:
            log.maybeLog(self.logger, 'Using default activation for visible layer')
            self.visible_activation = defaults['visible_activation']
            
        # Cost function!
        if args.get('cost_function') is not None:
            log.maybeLog(self.logger, '\nUsing specified cost function for GSN training\n')
            self.cost_function = args.get('cost_function')
        elif args.get('cost_funct') is not None:
            self.cost_function = get_cost_function(args.get('cost_funct'))
            log.maybeLog(self.logger, 'Using {0!s} for cost function'.format(args.get('cost_funct')))
        else:
            log.maybeLog(self.logger, '\nUsing default cost function for GSN training\n')
            self.cost_function = defaults['cost_function']
        
        ############################
        # Theano variables and RNG #
        ############################
        self.X = T.fmatrix('X') #single (batch) for training gsn
        self.Xs = T.fmatrix('Xs') #sequence for training rnn-gsn
        self.MRG = RNG_MRG.MRG_RandomStreams(1)
        
        ###############
        # Parameters! #
        ###############
        #gsn
        self.weights_list = [get_shared_weights(self.layer_sizes[i], self.layer_sizes[i+1], name="W_{0!s}_{1!s}".format(i,i+1)) for i in range(self.layers)] # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
        self.bias_list    = [get_shared_bias(self.layer_sizes[i], name='b_'+str(i)) for i in range(self.layers + 1)] # initialize each layer to 0's.
        
        #recurrent
        self.recurrent_to_gsn_weights_list = [get_shared_weights(self.recurrent_hidden_size, self.layer_sizes[layer], name="W_u_h{0!s}".format(layer)) for layer in range(self.layers+1) if layer%2 != 0]
        self.W_u_u = get_shared_weights(self.recurrent_hidden_size, self.recurrent_hidden_size, name="W_u_u")
        self.W_x_u = get_shared_weights(self.N_input, self.recurrent_hidden_size, name="W_x_u")
        self.recurrent_bias = get_shared_bias(self.recurrent_hidden_size, name='b_u')
        
        #lists for use with gradients
        self.gsn_params = self.weights_list + self.bias_list
        self.u_params   = [self.W_u_u, self.W_x_u, self.recurrent_bias]
        self.params     = self.gsn_params + self.recurrent_to_gsn_weights_list + self.u_params
        
        ###########################################################
        #           load initial parameters of gsn                #
        ###########################################################
        self.train_gsn_first = False
        if self.initialize_gsn:
            params_to_load = 'gsn_params.pkl'
            if not os.path.isfile(params_to_load):
                self.train_gsn_first = True 
            else:
                log.maybeLog(self.logger, "\nLoading existing GSN parameters\n")
                loaded_params = cPickle.load(open(params_to_load,'r'))
                [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[:len(self.weights_list)], self.weights_list)]
                [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[len(self.weights_list):], self.bias_list)]
                
        if self.initialize_gsn:
            self.gsn_args = {'weights_list':       self.weights_list,
                             'bias_list':          self.bias_list,
                             'hidden_activation':  self.hidden_activation,
                             'visible_activation': self.visible_activation,
                             'cost_function':      self.cost_function,
                             'layers':             self.layers,
                             'walkbacks':          self.walkbacks,
                             'hidden_size':        self.hidden_size,
                             'learning_rate':      args.get('learning_rate', defaults['learning_rate']),
                             'momentum':           args.get('momentum', defaults['momentum']),
                             'annealing':          self.annealing,
                             'noise_annealing':    self.noise_annealing,
                             'batch_size':         self.gsn_batch_size,
                             'n_epoch':            self.n_epoch,
                             'early_stop_threshold':   self.early_stop_threshold,
                             'early_stop_length':      self.early_stop_length,
                             'save_frequency':         self.save_frequency,
                             'noiseless_h1':           self.noiseless_h1,
                             'hidden_add_noise_sigma': args.get('hidden_add_noise_sigma', defaults['hidden_add_noise_sigma']),
                             'input_salt_and_pepper':  args.get('input_salt_and_pepper', defaults['input_salt_and_pepper']),
                             'input_sampling':      self.input_sampling,
                             'vis_init':            self.vis_init,
                             'output_path':         self.outdir+'gsn/',
                             'is_image':            self.is_image,
                             'input_size':          self.N_input
                             }
            
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
        gsn.update_layers(self.network_state_output,
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
    
               
        ##############################################
        #      Build the graphs for the RNN-GSN      #
        ##############################################
        # If `x_t` is given, deterministic recurrence to compute the u_t. Otherwise, first generate
        def recurrent_step(x_t, u_tm1, add_noise):
            # Make current guess for hiddens based on U
            for i in range(self.layers):
                if i%2 == 0:
                    log.maybeLog(self.logger, "Using {0!s} and {1!s}".format(self.recurrent_to_gsn_weights_list[(i+1)/2],self.bias_list[i+1]))
            h_t = T.concatenate([self.hidden_activation(self.bias_list[i+1] + T.dot(u_tm1, self.recurrent_to_gsn_weights_list[(i+1)/2])) for i in range(self.layers) if i%2 == 0],axis=0)
            
            generate = x_t is None
            if generate:
                pass
            
            # Make a GSN to update U
    #         chain, hs = gsn.build_gsn(x_t, weights_list, bias_list, add_noise, state.noiseless_h1, state.hidden_add_noise_sigma, state.input_salt_and_pepper, state.input_sampling, MRG, visible_activation, hidden_activation, walkbacks, logger)
    #         htop_t = hs[-1]
    #         denoised_x_t = chain[-1]
            # Update U
    #         ua_t = T.dot(denoised_x_t, W_x_u) + T.dot(htop_t, W_h_u) + T.dot(u_tm1, W_u_u) + recurrent_bias
            ua_t = T.dot(x_t, self.W_x_u) + T.dot(u_tm1, self.W_u_u) + self.recurrent_bias
            u_t = self.recurrent_hidden_activation(ua_t)
            return None if generate else [ua_t, u_t, h_t]
        
        log.maybeLog(self.logger, "\nCreating recurrent step scan.")
        # For training, the deterministic recurrence is used to compute all the
        # {h_t, 1 <= t <= T} given Xs. Conditional GSNs can then be trained
        # in batches using those parameters.
        u0 = T.zeros((self.recurrent_hidden_size,))  # initial value for the RNN hidden units
        (ua, u, h_t), updates_recurrent = theano.scan(fn=lambda x_t, u_tm1, *_: recurrent_step(x_t, u_tm1, True),
                                                           sequences=self.Xs,
                                                           outputs_info=[None, u0, None],
                                                           non_sequences=self.params)
        
        log.maybeLog(self.logger, "Now for reconstruction sample without noise")
        (_, _, h_t_recon), updates_recurrent_recon = theano.scan(fn=lambda x_t, u_tm1, *_: recurrent_step(x_t, u_tm1, False),
                                                           sequences=self.Xs,
                                                           outputs_info=[None, u0, None],
                                                           non_sequences=self.params)
        # put together the hiddens list
        h_list = [T.zeros_like(self.Xs)]
        for layer, w in enumerate(self.weights_list):
            if layer%2 != 0:
                h_list.append(T.zeros_like(T.dot(h_list[-1], w)))
            else:
                h_list.append((h_t.T[(layer/2)*self.hidden_size:(layer/2+1)*self.hidden_size]).T)
                
        h_list_recon = [T.zeros_like(self.Xs)]
        for layer, w in enumerate(self.weights_list):
            if layer%2 != 0:
                h_list_recon.append(T.zeros_like(T.dot(h_list_recon[-1], w)))
            else:
                h_list_recon.append((h_t_recon.T[(layer/2)*self.hidden_size:(layer/2+1)*self.hidden_size]).T)
        
        #with noise
        _, cost, show_cost = gsn.build_gsn_given_hiddens(self.Xs, h_list, self.weights_list, self.bias_list, True, self.noiseless_h1, self.hidden_add_noise_sigma, self.input_salt_and_pepper, self.input_sampling, self.MRG, self.visible_activation, self.hidden_activation, self.walkbacks, self.cost_function, self.logger)
        #without noise for reconstruction
        x_sample_recon, _, _ = gsn.build_gsn_given_hiddens(self.Xs, h_list_recon, self.weights_list, self.bias_list, False, self.noiseless_h1, self.hidden_add_noise_sigma, self.input_salt_and_pepper, self.input_sampling, self.MRG, self.visible_activation, self.hidden_activation, self.walkbacks, self.cost_function, self.logger)
        
        updates_train = updates_recurrent
        updates_cost = updates_recurrent
        
        #############
        #   COSTS   #
        #############
        log.maybeLog(self.logger, '\nCost w.r.t p(X|...) at every step in the graph')
        start_functions_time = time.time()

        # if we are not using Hessian-free training create the normal sgd functions
        if not self.hessian_free:
            gradient      = T.grad(cost, self.params)      
            gradient_buffer = [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in self.params]
            
            m_gradient    = [self.momentum * gb + (cast32(1) - self.momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
            param_updates = [(param, param - self.learning_rate * mg) for (param, mg) in zip(self.params, m_gradient)]
            gradient_buffer_updates = zip(gradient_buffer, m_gradient)
                
            updates = OrderedDict(param_updates + gradient_buffer_updates)
            updates_train.update(updates)
        
            log.maybeLog(self.logger, "rnn-gsn learn...")
            self.f_learn = theano.function(inputs  = [self.Xs],
                                      updates = updates_train,
                                      outputs = show_cost,
                                      on_unused_input='warn',
                                      name='rnngsn_f_learn')
            
            log.maybeLog(self.logger, "rnn-gsn cost...")
            self.f_cost  = theano.function(inputs  = [self.Xs],
                                      updates = updates_cost,
                                      outputs = show_cost, 
                                      on_unused_input='warn',
                                      name='rnngsn_f_cost')
        
        log.maybeLog(self.logger, "Training/cost functions done.")
        
        # Denoise some numbers : show number, noisy number, predicted number, reconstructed number
        log.maybeLog(self.logger, "Creating graph for noisy reconstruction function at checkpoints during training.")
        self.f_recon = theano.function(inputs=[self.Xs],
                                       outputs=x_sample_recon[-1],
                                       name='rnngsn_f_recon')
        
        # a function to add salt and pepper noise
        self.f_noise = theano.function(inputs = [self.X],
                                       outputs = salt_and_pepper(self.X, self.input_salt_and_pepper),
                                       name='rnngsn_f_noise')
        # Sampling functions
        log.maybeLog(self.logger, "Creating sampling function...")
        if self.layers == 1: 
            self.f_sample = theano.function(inputs = [X_sample],
                                            outputs = visible_pX_chain[-1],
                                            name='rnngsn_f_sample_single_layer')
        else:
            self.f_sample = theano.function(inputs = self.network_state_input,
                                            outputs = self.network_state_output + visible_pX_chain,
                                            on_unused_input='warn',
                                            name='rnngsn_f_sample')
        
    
        log.maybeLog(self.logger, "Done compiling all functions.")
        compilation_time = time.time() - start_functions_time
        # Show the compile time with appropriate easy-to-read units.
        log.maybeLog(self.logger, "Total compilation time took "+make_time_units_string(compilation_time)+".\n\n")
      
        
        
    def train(self, train_X=None, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None, is_artificial=False, artificial_sequence=1, continue_training=False):
        log.maybeLog(self.logger, "\nTraining---------\n")
        if train_X is None:
            log.maybeLog(self.logger, "Training using data given during initialization of RNN-GSN.\n")
            train_X = self.train_X
            train_Y = self.train_Y
            if train_X is None:
                log.maybeLog(self.logger, "\nPlease provide a training dataset!\n")
                raise AssertionError("Please provide a training dataset!")
        else:
            log.maybeLog(self.logger, "Training using data provided to training function.\n")
        if valid_X is None:
            valid_X = self.valid_X
            valid_Y = self.valid_Y
        if test_X is None:
            test_X  = self.test_X
            test_Y  = self.test_Y
            
        # Input data - make sure it is a list of shared datasets
        train_X = raise_to_list(train_X)
        train_Y = raise_to_list(train_Y)
        valid_X = raise_to_list(valid_X)
        valid_Y = raise_to_list(valid_Y)
        test_X  = raise_to_list(test_X)
        test_Y =  raise_to_list(test_Y)
            
        ##########################################################
        # Train the GSN first to get good weights initialization #
        ##########################################################
        if self.train_gsn_first:
            log.maybeLog(self.logger, "\n\n----------Initially training the GSN---------\n\n")
            init_gsn = gsn.GSN(train_X=train_X, valid_X=valid_X, test_X=test_X, args=self.gsn_args, logger=self.logger)
            init_gsn.train()
    
        
        #########################################
        # If we are using Hessian-free training #
        #########################################
        if self.hessian_free:
            pass
#         gradient_dataset = hf_sequence_dataset([train_X.get_value()], batch_size=None, number_batches=5000)
#         cg_dataset = hf_sequence_dataset([train_X.get_value()], batch_size=None, number_batches=1000)
#         valid_dataset = hf_sequence_dataset([valid_X.get_value()], batch_size=None, number_batches=1000)
#         
#         s = x_samples
#         costs = [cost, show_cost]
#         hf_optimizer(params, [Xs], s, costs, u, ua).train(gradient_dataset, cg_dataset, initial_lambda=1.0, preconditioner=True, validation=valid_dataset)
        
        ################################
        # If we are using SGD training #
        ################################
        else:
            log.maybeLog(self.logger, "\n-----------TRAINING RNN-GSN------------\n")
            # TRAINING
            STOP        =   False
            counter     =   0
            if not continue_training:
                self.learning_rate.set_value(self.init_learn_rate)  # learning rate
            times = []
            best_cost = float('inf')
            best_params = None
            patience = 0
                        
            log.maybeLog(self.logger, ['train X size:',str(train_X[0].shape.eval())])
            if valid_X is not None:
                log.maybeLog(self.logger, ['valid X size:',str(valid_X[0].shape.eval())])
            if test_X is not None:
                log.maybeLog(self.logger, ['test X size:',str(test_X[0].shape.eval())])
            
            if self.vis_init:
                self.bias_list[0].set_value(logit(numpy.clip(0.9,0.001,train_X.get_value().mean(axis=0))))
        
            while not STOP:
                counter += 1
                t = time.time()
                log.maybeAppend(self.logger, [counter,'\t'])
                    
                if is_artificial:
                    data.sequence_mnist_data(train_X[0], train_Y[0], valid_X[0], valid_Y[0], test_X[0], test_Y[0], artificial_sequence, rng)
                     
                #train
                train_costs = []
                for train_data in train_X:
                    train_costs.extend(data.apply_cost_function_to_dataset(self.f_learn, train_data, self.batch_size))
                log.maybeAppend(self.logger, ['Train:',trunc(numpy.mean(train_costs)),'\t'])
         
         
                #valid
                if valid_X is not None:
                    valid_costs = []
                    for valid_data in valid_X:
                        valid_costs.extend(data.apply_cost_function_to_dataset(self.f_cost, valid_data, self.batch_size))
                    log.maybeAppend(self.logger, ['Valid:',trunc(numpy.mean(valid_costs)), '\t'])
         
         
                #test
                if test_X is not None:
                    test_costs = []
                    for test_data in test_X:
                        test_costs.extend(data.apply_cost_function_to_dataset(self.f_cost, test_data, self.batch_size))
                    log.maybeAppend(self.logger, ['Test:',trunc(numpy.mean(test_costs)), '\t'])
                
                 
                #check for early stopping
                if valid_X is not None:
                    cost = numpy.sum(valid_costs)
                else:
                    cost = numpy.sum(train_costs)
                if cost < best_cost*self.early_stop_threshold:
                    patience = 0
                    best_cost = cost
                    # save the parameters that made it the best
                    best_params = copy_params(self.params)
                else:
                    patience += 1
         
                if counter >= self.n_epoch or patience >= self.early_stop_length:
                    STOP = True
                    if best_params is not None:
                        restore_params(self.params, best_params)
                    self.save_params('all', counter, self.params)
         
                timing = time.time() - t
                times.append(timing)
         
                log.maybeAppend(self.logger, 'time: '+make_time_units_string(timing)+'\t')
            
                log.maybeLog(self.logger, 'remaining: '+make_time_units_string((self.n_epoch - counter) * numpy.mean(times)))
        
                if (counter % self.save_frequency) == 0 or STOP is True:
                    if (self.is_image):
                        n_examples = 100
                        xs_test = test_X[0].get_value(borrow=True)[range(n_examples)]
                        noisy_xs_test = self.f_noise(test_X[0].get_value(borrow=True)[range(n_examples)])
                        reconstructions = []
                        for i in xrange(0, len(noisy_xs_test)):
                            recon = self.f_recon(noisy_xs_test[max(0,(i+1)-self.batch_size):i+1])
                            reconstructions.append(recon)
                        reconstructed = numpy.array(reconstructions)
    
                        # Concatenate stuff
                        stacked = numpy.vstack([numpy.vstack([xs_test[i*10 : (i+1)*10], noisy_xs_test[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
                        number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (self.image_height, self.image_width), (10,30)))
                            
                        number_reconstruction.save(self.outdir+'rnngsn_reconstruction_epoch_'+str(counter)+'.png')
            
                        #sample_numbers(counter, 'seven')
#                         plot_samples(counter, 'rnngsn')
            
                    #save params
                    self.save_params('all', counter, self.params)
             
                # ANNEAL!
                new_lr = self.learning_rate.get_value() * self.annealing
                self.learning_rate.set_value(new_lr)
                
                new_noise = self.input_salt_and_pepper.get_value() * self.noise_annealing
                self.input_salt_and_pepper.set_value(new_noise)
    
            
            # 10k samples
#             print 'Generating 10,000 samples'
#             samples, _  =   sample_some_numbers(N=10000)
#             f_samples   =   self.outdir+'samples.npy'
#             numpy.save(f_samples, samples)
#             print 'saved digits'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
        img_samples = PIL.Image.fromarray(tile_raster_images(V, (self.image_height, self.image_width), (ceil(sqrt(n_samples)), ceil(sqrt(n_samples)))))
        
        fname = self.outdir+leading_text+'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        log.maybeLog(self.logger, 'Took ' + str(time.time() - to_sample) + ' to sample '+n_samples+' numbers')
        
    #############################
    # Save the model parameters #
    #############################                       
    def save_params(self, name, n, params):
        log.maybeLog(self.logger, 'saving parameters...')
        save_path = self.outdir+name+'_params_epoch_'+str(n)+'.pkl'
        f = open(save_path, 'wb')
        try:
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close()
            
    def load_params(self, filename):
        '''
        self.params = self.weights_list + self.bias_list + self.recurrent_to_gsn_weights_list + [self.W_u_u, self.W_x_u, self.recurrent_bias]
        '''
        def set_param(loaded_params, start, param):
            [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(loaded_params[start:start+len(param)], param)]
            return start + len(param)
            
        if os.path.isfile(filename):
            log.maybeLog(self.logger, "\nLoading existing RNN-GSN parameters")
            loaded_params = cPickle.load(open(filename,'r'))
            start = 0
            start = set_param(loaded_params, start, self.weights_list)
            start = set_param(loaded_params, start, self.bias_list)
            start = set_param(loaded_params, start, self.recurrent_to_gsn_weights_list)
            set_param(loaded_params, start, self.u_params)
        else:
            log.maybeLog(self.logger, "\n\nCould not find existing RNN-GSN parameter file {}.\n\n".format(filename))
        
                
    
    
    
