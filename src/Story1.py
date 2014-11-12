import numpy, os, cPickle
import numpy.random as rng
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import tile_raster_images
import time
import data_tools as data
from utils import cast32, trunc, logit, get_shared_weights, get_shared_bias, get_shared_regression_weights, add_gaussian_noise, salt_and_pepper, load_from_config, fix_input_size

def experiment(state, outdir_base='./'):
    rng.seed(1) #seed the numpy random generator  
    # Initialize the output directories and files
    data.mkdir_p(outdir_base)
    outdir = outdir_base + "/" + state.dataset + "/"
    data.mkdir_p(outdir)
    logfile = outdir+"log.txt"
    with open(logfile,'w') as f:
        f.write("MODEL 1, {0!s}\n\n".format(state.dataset))
    train_convergence = outdir+"train_convergence.csv"
    valid_convergence = outdir+"valid_convergence.csv"
    test_convergence = outdir+"test_convergence.csv"
    regression_train_convergence = outdir+"regression_train_convergence.csv"
    regression_valid_convergence = outdir+"regression_valid_convergence.csv"
    regression_test_convergence = outdir+"regression_test_convergence.csv"

    print
    print "----------MODEL 1, {0!s}--------------".format(state.dataset)
    print
    #load parameters from config file if this is a test
    config_filename = outdir+'config'
    if state.test_model and 'config' in os.listdir(outdir):
        config_vals = load_from_config(config_filename)
        for CV in config_vals:
            print CV
            if CV.startswith('test'):
                print 'Do not override testing switch'
                continue        
            try:
                exec('state.'+CV) in globals(), locals()
            except:
                exec('state.'+CV.split('=')[0]+"='"+CV.split('=')[1]+"'") in globals(), locals()
    else:
        # Save the current configuration
        # Useful for logs/experiments
        print 'Saving config'
        with open(config_filename, 'w') as f:
            f.write(str(state))


    print state
    artificial = False #internal flag to see if the dataset is one of my artificially-sequenced MNIST varieties.
    if state.dataset == 'MNIST_1' or state.dataset == 'MNIST_2' or state.dataset == 'MNIST_3':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))
        artificial = True
        try:
            dataset = int(state.dataset.split('_')[1])
        except:
            raise AssertionError("artificial dataset number not recognized. Input was "+state.dataset)
    else:
        raise AssertionError("dataset not recognized.")

    # transfer the datasets into theano shared variables
    train_X = theano.shared(train_X)
    train_Y = theano.shared(train_Y)
    valid_X = theano.shared(valid_X)
    valid_Y = theano.shared(valid_Y) 
    test_X = theano.shared(test_X)
    test_Y = theano.shared(test_Y) 
    
    if artificial: #if it my MNIST sequence, appropriately sequence it.
        print 'Sequencing MNIST data...'
        print 'train set size:',len(train_Y.eval())
        print 'valid set size:',len(valid_Y.eval())
        print 'test set size:',len(test_Y.eval())
        data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset, rng)
        print 'train set size:',len(train_Y.eval())
        print 'valid set size:',len(valid_Y.eval())
        print 'test set size:',len(test_Y.eval())
        print 'Sequencing done.'
        print
    
    # varaibles from the dataset that are used for initialization and image reconstruction
    N_input =   train_X.eval().shape[1]
    root_N_input = numpy.sqrt(N_input)
    
    # Network and training specifications
    layers                   = state.layers # number hidden layers
    walkbacks                = state.walkbacks # number of walkbacks
    sequence_window_size     = state.sequence_window_size # number of previous hidden states to consider for the regression
    layer_sizes              = [N_input] + [state.hidden_size] * layers # layer sizes, from h0 to hK (h0 is the visible layer)
    learning_rate            = theano.shared(cast32(state.learning_rate))  # learning rate
    regression_learning_rate = theano.shared(cast32(state.learning_rate))  # learning rate
    annealing                = cast32(state.annealing) # exponential annealing coefficient
    momentum                 = theano.shared(cast32(state.momentum)) # momentum term 
    
    # Theano variables and RNG
    X   = T.fmatrix('X') # for use in sampling
    Xs  = [T.fmatrix(name="X_t") if i==0 else T.fmatrix(name="X_{t-"+str(i)+"}") for i in range(sequence_window_size+1)] # for use in training - need one X variable for each input in the sequence history window, and what the current one should be
    Xs_recon  = [T.fvector(name="Xrecon_t") if i==0 else T.fvector(name="Xrecon_{t-"+str(i)+"}") for i in range(sequence_window_size+1)] # for use in training - need one X variable for each input in the sequence history window, and what the current one should be
    MRG = RNG_MRG.MRG_RandomStreams(1)

    # PARAMETERS : weights list and bias list.
    # initialize a list of weights and biases based on layer_sizes for the GSN
    weights_list = [get_shared_weights(layer_sizes[layer], layer_sizes[layer+1], name="W_{0!s}_{1!s}".format(layer,layer+1)) for layer in range(layers)] # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
    bias_list    = [get_shared_bias(layer_sizes[layer], name='b_'+str(layer)) for layer in range(layers + 1)] # initialize each layer to 0's.
    # parameters for the regression - only need them for the odd layers in the network!
    regression_weights_list = [[get_shared_regression_weights(state.hidden_size, name="V_{t-"+str(window+1)+"}_layer"+str(layer)) for layer in range(layers+1) if (layer%2) != 0] for window in range(sequence_window_size)] # initialize to identity matrix the size of hidden layer.
    regression_bias_list    = [get_shared_bias(state.hidden_size, name='vb_'+str(layer)) for layer in range(layers+1) if (layer%2) != 0] # initialize to 0's. 
    #need initial biases (tau) as well for when there aren't sequence_window_size hiddens in the history.
    tau_list                = [[get_shared_bias(state.hidden_size, name='tau_{t-'+str(window+1)+"}_layer"+str(layer)) for layer in range(layers+1) if (layer%2) != 0] for window in range(sequence_window_size)]

#     if state.test_model:
#         # Load the parameters of the last epoch
#         # maybe if the path is given, load these specific attributes 
#         param_files     =   filter(lambda x:'gsn_params' in x, os.listdir('.'))
#         max_epoch_idx   =   numpy.argmax([int(x.split('_')[-1].split('.')[0]) for x in param_files])
#         params_to_load  =   param_files[max_epoch_idx]
#         PARAMS = cPickle.load(open(params_to_load,'r'))
#         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
#         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]
#         
#     if state.continue_training:
#         # Load the parameters of the last GSN
#         params_to_load = 'gsn_params.pkl'
#         PARAMS = cPickle.load(open(params_to_load,'r'))
#         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
#         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]
#         # Load the parameters of the last recurrent
# #         params_to_load = 'regression_params.pkl'
# #         PARAMS = cPickle.load(open(params_to_load,'r'))
# #         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
# #         [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]

 
    ''' F PROP '''
    if state.hidden_act == 'sigmoid':
        print 'Using sigmoid activation for hiddens'
        hidden_activation = T.nnet.sigmoid
    elif state.hidden_act == 'rectifier':
        print 'Using rectifier activation for hiddens'
        hidden_activation = lambda x : T.maximum(cast32(0), x)
    elif state.hidden_act == 'tanh':
        print 'Using hyperbolic tangent activation for hiddens'
        hidden_activation = lambda x : T.tanh(x)
    else:
        raise AssertionError("Did not recognize hidden activation {0!s}, please use tanh, rectifier, or sigmoid".format(state.hidden_act))
    
    if state.visible_act == 'sigmoid':
        print 'Using sigmoid activation for visible layer'
        visible_activation = T.nnet.sigmoid
    elif state.visible_act == 'softmax':
        print 'Using softmax activation for visible layer'
        visible_activation = T.nnet.softmax
    else:
        raise AssertionError("Did not recognize visible activation {0!s}, please use sigmoid or softmax".format(state.visible_act))
  
  
    ###############################################
    # COMPUTATIONAL GRAPH HELPER METHODS FOR TGSN #
    ###############################################
    def update_layers(hiddens, p_X_chain, noisy = True):
        print 'odd layer updates'
        update_odd_layers(hiddens, noisy)
        print 'even layer updates'
        update_even_layers(hiddens, p_X_chain, noisy)
        print 'done full update.'
        print
        
    def update_layers_reverse(hiddens, p_X_chain, noisy = True):
        print 'even layer updates'
        update_even_layers(hiddens, p_X_chain, noisy)
        print 'odd layer updates'
        update_odd_layers(hiddens, noisy)
        print 'done full update.'
        print
        
    # Odd layer update function
    # just a loop over the odd layers
    def update_odd_layers(hiddens, noisy):
        for i in range(1, len(hiddens), 2):
            print 'updating layer',i
            simple_update_layer(hiddens, None, i, add_noise = noisy)
    
    # Even layer update
    # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
    def update_even_layers(hiddens, p_X_chain, noisy):
        for i in range(0, len(hiddens), 2):
            print 'updating layer',i
            simple_update_layer(hiddens, p_X_chain, i, add_noise = noisy)
    
    # The layer update function
    # hiddens   :   list containing the symbolic theano variables [visible, hidden1, hidden2, ...]
    #               layer_update will modify this list inplace
    # p_X_chain :   list containing the successive p(X|...) at each update
    #               update_layer will append to this list
    # i         :   the current layer being updated
    # add_noise :   pre (and post) activation gaussian noise flag
    def simple_update_layer(hiddens, p_X_chain, i, add_noise=True):   
        # Compute the dot product, whatever layer
        # If the visible layer X
        if i == 0:
            print 'using '+str(weights_list[i])+'.T'
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]           
        # If the top layer
        elif i == len(hiddens)-1:
            print 'using',weights_list[i-1]
            hiddens[i]  =   T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
        # Otherwise in-between layers
        else:
            print "using {0!s} and {1!s}.T".format(weights_list[i-1], weights_list[i])
            # next layer        :   hiddens[i+1], assigned weights : W_i
            # previous layer    :   hiddens[i-1], assigned weights : W_(i-1)
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
    
        # Add pre-activation noise if NOT input layer
        if i==1 and state.noiseless_h1:
            print '>>NO noise in first hidden layer'
            add_noise   =   False
    
        # pre activation noise            
        if i != 0 and add_noise:
            print 'Adding pre-activation gaussian noise for layer', i
            hiddens[i] = add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
       
        # ACTIVATION!
        if i == 0:
            print '{} activation for visible layer'.format(state.visible_act)
            hiddens[i] = visible_activation(hiddens[i])
        else:
            print 'Hidden units {} activation for layer'.format(state.hidden_act), i
            hiddens[i] = hidden_activation(hiddens[i])
    
        # post activation noise
        # why is there post activation noise? Because there is already pre-activation noise, this just doubles the amount of noise between each activation of the hiddens.  
#         if i != 0 and add_noise:
#             print 'Adding post-activation gaussian noise for layer', i
#             hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
    
        # build the reconstruction chain if updating the visible layer X
        if i == 0:
            # if input layer -> append p(X|...)
            p_X_chain.append(hiddens[i])
            
            # sample from p(X|...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
            if state.input_sampling:
                print 'Sampling from input'
                sampled = MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
                print '>>NO input sampling'
                sampled = hiddens[i]
            # add noise
            sampled = salt_and_pepper(sampled, state.input_salt_and_pepper)
            
            # set input layer
            hiddens[i] = sampled
                
    def perform_regression_step(hiddens, sequence_history):
        print "Sequence history length:",len(sequence_history)
        # only need to work over the odd layers of the hiddens
        odd_layers = [i for i in range(len(hiddens)) if (i%2) != 0]
        # depending on the size of the sequence history, it could be 0, 1, 2, 3, ... sequence_window_size
        for (hidden_index,regression_index) in zip(odd_layers,range(len(odd_layers))):
            terms_used = []
            sequence_terms = []
            for history_index in range(sequence_window_size):
                if history_index < len(sequence_history):
                    # dot product with history term
                    sequence_terms.append(T.dot(sequence_history[history_index][regression_index],regression_weights_list[history_index][regression_index]))
                    terms_used.append(regression_weights_list[history_index][regression_index])
                else:
                    # otherwise, no history for necessary spot, so use the tau
                    sequence_terms.append(tau_list[history_index][regression_index])
                    terms_used.append(tau_list[history_index][regression_index])
                    
            sequence_terms.append(regression_bias_list[regression_index])
            terms_used.append(regression_bias_list[regression_index])
            print "REGRESSION for hidden layer {0!s} using:".format(hidden_index), terms_used
            hiddens[hidden_index] = numpy.sum(sequence_terms)
    
    def build_gsn_graph(x, noiseflag):
        p_X_chain = []
        if noiseflag:
            X_init = salt_and_pepper(x, state.input_salt_and_pepper)
        else:
            X_init = x
        # init hiddens with zeros
        hiddens = [X_init]
        for w in weights_list:
            hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
            
        # The layer update scheme
        print "Building the gsn graph :", walkbacks,"updates"
        for i in range(walkbacks):
            print "GSN Walkback {!s}/{!s}".format(i+1,walkbacks)
            update_layers(hiddens, p_X_chain, noisy=noiseflag)
            
        return p_X_chain
            
    
    def build_sequence_graph(xs, noiseflag):
        predicted_X_chains = []
        p_X_chains = []
        sequence_history = []
        # The layer update scheme
        print "Building the regression graph :", len(Xs),"updates"
        ''' hidden layer init '''
        hiddens = [T.zeros_like(xs[0])]
        for w in weights_list:
            # init with zeros
            hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))        
        for x in xs:
            print "Performing regression step!"
            perform_regression_step(hiddens, sequence_history) # do the regression!
            print ""
            # Predict what the current X should be
            pred_X_chain = []
            pred_hiddens = hiddens
            for i in range(walkbacks):
                print "Prediction Walkback {!s}/{!s}".format(i+1,walkbacks)
                update_layers_reverse(pred_hiddens, pred_X_chain, noisy=noiseflag)
            # Append the predicted X
            predicted_X_chains.append(pred_X_chain)
            
            # Now use the actual X as hiddens[0]
            # Need to advance the regression by 1 update
            print "moving hiddens forward one half update."
            update_even_layers(hiddens, [], noisy=noiseflag)
            print
            # corrupt x if noisy
            if noiseflag:
                X_init = salt_and_pepper(x, state.input_salt_and_pepper)
            else:
                X_init = x
            hiddens[0] = X_init
            p_X_chain = []
            for i in range(walkbacks):
                print "Actual walkback {!s}/{!s}".format(i+1,walkbacks)
                update_layers(hiddens, p_X_chain, noisy=noiseflag)
            # Append the p_X_chain
            p_X_chains.append(p_X_chain)
            
            # Append the odd layers of the hiddens to the sequence history
            sequence_history.append([hiddens[layer] for layer in range(len(hiddens)) if (layer%2) != 0])
        
        return predicted_X_chains, p_X_chains
            
        
    ##############################################
    #    Build the training graph for the GSN    #
    ##############################################
    print
    print "Building GSN graphs"
    p_X_chain = build_gsn_graph(X, noiseflag=True)
    _, p_X_chains = build_sequence_graph(Xs, noiseflag=True)
    
    ###############################################
    # Build the training graph for the regression #
    ###############################################
    print
    print "Building regression graph"
    # no noise! noise is only used as regularization for GSN stage
    predicted_X_chains, _ = build_sequence_graph(Xs, noiseflag=False)


    ######################
    # COST AND GRADIENTS #
    ######################
    print
    print 'Cost w.r.t p(X|...) at every step in the graph for the TGSN'
    gsn_costs_init     = [T.mean(T.nnet.binary_crossentropy(rX, X)) for rX in p_X_chain]
    show_gsn_cost_init = gsn_costs_init[-1]
    gsn_cost_init      = numpy.sum(gsn_costs_init)
    
    gsn_costs      = [[T.mean(T.nnet.binary_crossentropy(rX, Xs[i])) for rX in p_X_chains[i]] for i in range(len(Xs))]
    show_gsn_costs = [costs[-1] for costs in gsn_costs]
    gsn_cost       = numpy.sum([numpy.sum(costs) for costs in gsn_costs])
    
    gsn_params = weights_list + bias_list    
    print "gsn params:",gsn_params
        
    
    #l2 regularization
    #regression_regularization_cost = T.sum([T.sum(recurrent_weights ** 2) for recurrent_weights in regression_weights_list])
    regression_regularization_cost = 0
    #regression_cost = T.log(T.sum(T.pow((predicted_network - encoded_network),2)) + regression_regularization_cost)
    regression_costs = [[T.mean(T.nnet.binary_crossentropy(rX, Xs[i])) for rX in predicted_X_chains[i]] for i in range(len(Xs))]
    show_regression_costs = [costs[-1] for costs in regression_costs]
    regression_cost = numpy.sum(regression_costs) + state.regularize_weight * regression_regularization_cost
    
    #only using the odd layers update -> even-indexed parameters in the list because it starts at v1
    # need to flatten the regression and tau lists -> couldn't immediately find the python method so here is the implementation
    regression_weights_flattened = []
    for weights in regression_weights_list:
        regression_weights_flattened.extend(weights)
    tau_flattened = []
    for tau in tau_list:
        tau_flattened.extend(tau)
    regression_params = regression_weights_flattened + regression_bias_list + tau_flattened
    print "regression params:", regression_params    
    
    
    
    print "creating functions..."
    t = time.time()
    
    gradient_init        =   T.grad(gsn_cost_init, gsn_params)              
    gradient_buffer_init =   [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in gsn_params] 
    m_gradient_init      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer_init, gradient_init)]
    param_updates_init   =   [(param, param - learning_rate * mg) for (param, mg) in zip(gsn_params, m_gradient_init)]
    gradient_buffer_updates_init = zip(gradient_buffer_init, m_gradient_init)
    updates_init         =   OrderedDict(param_updates_init + gradient_buffer_updates_init)
    
    
    gsn_f_learn_init     =   theano.function(inputs  = [X], 
                                         updates = updates_init, 
                                         outputs = show_gsn_cost_init)
    
    gsn_f_cost_init      =   theano.function(inputs  = [X], 
                                             outputs = show_gsn_cost_init)
    
    
    gradient        =   T.grad(gsn_cost, gsn_params)
    gradient_buffer =   [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in gsn_params]
    m_gradient      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
    param_updates   =   [(param, param - learning_rate * mg) for (param, mg) in zip(gsn_params, m_gradient)]
    gradient_buffer_updates = zip(gradient_buffer, m_gradient)
        
    updates         =   OrderedDict(param_updates + gradient_buffer_updates)
        
    
    gsn_f_cost           =   theano.function(inputs  = Xs, 
                                         outputs = show_gsn_costs)


    gsn_f_learn          =   theano.function(inputs  = Xs, 
                                         updates = updates, 
                                         outputs = show_gsn_costs)
      
    
    regression_gradient        =   T.grad(regression_cost, regression_params)
    regression_gradient_buffer =   [theano.shared(numpy.zeros(rparam.get_value().shape, dtype='float32')) for rparam in regression_params]
    regression_m_gradient      =   [momentum * rgb + (cast32(1) - momentum) * rg for (rgb, rg) in zip(regression_gradient_buffer, regression_gradient)]
    regression_param_updates   =   [(rparam, rparam - regression_learning_rate * rmg) for (rparam, rmg) in zip(regression_params, regression_m_gradient)]
    regression_gradient_buffer_updates = zip(regression_gradient_buffer, regression_m_gradient)
        
    regression_updates         =   OrderedDict(regression_param_updates + regression_gradient_buffer_updates)
    
    regression_f_cost          =   theano.function(inputs = Xs, 
                                                   outputs = show_regression_costs)
        
    regression_f_learn         =   theano.function(inputs  = Xs, 
                                                  updates = regression_updates, 
                                                  outputs = show_regression_costs)
    
    print "functions done. took {0!s} seconds.".format(trunc(time.time() - t))
    print
    
    
    ############################################################################################
    # Denoise some numbers : show number, noisy number, predicted number, reconstructed number #
    ############################################################################################   
    # Recompile the graph without noise for reconstruction function
    # The layer update scheme
    print "Creating graph for noisy reconstruction function at checkpoints during training."
    predicted_X_chains_R, p_X_chains_R = build_sequence_graph(Xs_recon, noiseflag=False)
    predictions_R = [pchain[-1] for pchain in predicted_X_chains_R]
    reconstructions_R = [pchain[-1] for pchain in p_X_chains_R]
    f_recon = theano.function(inputs = Xs_recon, outputs = predictions_R + reconstructions_R)


    ############
    # Sampling #
    ############
    f_noise = theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))
    # the input to the sampling function
    network_state_input     =   [X] + [T.fmatrix() for i in range(layers)]
   
    # "Output" state of the network (noisy)
    # initialized with input, then we apply updates
    #network_state_output    =   network_state_input
    
    network_state_output    =   [X] + network_state_input[1:]

    visible_pX_chain        =   []

    # ONE update
    print "Performing one walkback in network state sampling."
    update_layers(network_state_output, visible_pX_chain, noisy=True)

    if layers == 1: 
        f_sample_simple = theano.function(inputs = [X], outputs = visible_pX_chain[-1])
    
    
    # WHY IS THERE A WARNING????
    # because the first odd layers are not used -> directly computed FROM THE EVEN layers
    # unused input = warn
    f_sample2   =   theano.function(inputs = network_state_input, outputs = network_state_output + visible_pX_chain, on_unused_input='warn')

    def sample_some_numbers_single_layer():
        x0    =   test_X.get_value()[7:8]
        samples = [x0]
        x  =   f_noise(x0)
        for i in range(399):
            x = f_sample_simple(x)
            samples.append(x)
            x = numpy.random.binomial(n=1, p=x, size=x.shape).astype('float32')
            x = f_noise(x)
        return numpy.vstack(samples)
            
    def sampling_wrapper(NSI):
        # * is the "splat" operator: It takes a list as input, and expands it into actual positional arguments in the function call.
        out             =   f_sample2(*NSI)
        NSO             =   out[:len(network_state_output)]
        vis_pX_chain    =   out[len(network_state_output):]
        return NSO, vis_pX_chain

    def sample_some_numbers(N=400):
        # The network's initial state
        init_vis        =   test_X.get_value()[7:8]

        noisy_init_vis  =   f_noise(init_vis)

        network_state   =   [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [noisy_init_vis]

        for i in range(N-1):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain = sampling_wrapper(network_state[-1])

            # append to the visible chain
            visible_chain += vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
    
    def plot_samples(epoch_number, iteration):
        to_sample = time.time()
        if layers == 1:
            # one layer model
            V = sample_some_numbers_single_layer()
        else:
            V, _ = sample_some_numbers()
        img_samples =   PIL.Image.fromarray(tile_raster_images(V, (root_N_input,root_N_input), (20,20)))
        
        fname       =   outdir+'samples_iteration_'+str(iteration)+'_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        print 'Took ' + str(time.time() - to_sample) + ' to sample 400 numbers'
   
   
    #############################
    # Save the model parameters #
    #############################
    def save_params(name, n, gsn_params, iteration):
        print 'saving parameters...'
        save_path = outdir+name+'_params_iteration_'+str(iteration)+'_epoch_'+str(n)+'.pkl'
        f = open(save_path, 'wb')
        try:
            cPickle.dump(gsn_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close() 


    ################
    # GSN TRAINING #
    ################
    def train_GSN(iteration, train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
        print '----------------------------------------'
        print 'TRAINING GSN FOR ITERATION',iteration
        with open(logfile,'a') as f:
            f.write("--------------------------\nTRAINING GSN FOR ITERATION {0!s}\n".format(iteration))
        
        # TRAINING
        n_epoch     =   state.n_epoch
        batch_size  =   state.batch_size
        STOP        =   False
        counter     =   0
        if iteration == 0:
            learning_rate.set_value(cast32(state.learning_rate))  # learning rate
        times = []
        best_cost = float('inf')
        patience = 0
            
        print 'learning rate:',learning_rate.get_value()
        
        print 'train X size:',str(train_X.shape.eval())
        print 'valid X size:',str(valid_X.shape.eval())
        print 'test X size:',str(test_X.shape.eval())
        
        if state.vis_init:
            bias_list[0].set_value(logit(numpy.clip(0.9,0.001,train_X.get_value().mean(axis=0))))
    
        if state.test_model:
            # If testing, do not train and go directly to generating samples, parzen window estimation, and inpainting
            print 'Testing : skip training'
            STOP    =   True
    
    
        while not STOP:
            counter += 1
            t = time.time()
            print counter,'\t',
            with open(logfile,'a') as f:
                f.write("{0!s}\t".format(counter))
                
            #shuffle the data
            data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset, rng)
                
            #train
            train_costs = []
            if iteration == 0:
                for i in range(len(train_X.get_value(borrow=True)) / batch_size):
                    x = train_X.get_value()[i * batch_size : (i+1) * batch_size]
                    cost = gsn_f_learn_init(x)
                    train_costs.append([cost])
            else:
                for i in range(len(train_X.get_value(borrow=True)) / batch_size):
                    xs = [train_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                    xs, _ = fix_input_size(xs)
                    costs = gsn_f_learn(*xs)
                    train_costs.append(costs)
                
            train_costs = numpy.mean(train_costs, 0) 
            print 'Train : ',[trunc(cost) for cost in train_costs], '\t',
            with open(logfile,'a') as f:
                f.write("Train : {0!s}\t".format([trunc(cost) for cost in train_costs]))
            with open(train_convergence,'a') as f:
                for cost in train_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")
    
            #valid
            valid_costs  =  []
            if iteration == 0:
                for i in range(len(valid_X.get_value(borrow=True)) / batch_size):
                    x = valid_X.get_value()[i * batch_size : (i+1) * batch_size]
                    cost = gsn_f_cost_init(x)
                    valid_costs.append([cost])
            else:
                for i in range(len(valid_X.get_value(borrow=True)) / batch_size):
                    xs = [valid_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                    xs, _ = fix_input_size(xs)
                    costs = gsn_f_cost(*xs)
                    valid_costs.append(costs)
                    
            valid_costs = numpy.mean(valid_costs, 0) 
            print 'Valid : ', [trunc(cost) for cost in valid_costs], '\t',
            with open(logfile,'a') as f:
                f.write("Valid : {0!s}\t".format([trunc(cost) for cost in valid_costs]))
            with open(valid_convergence,'a') as f:
                for cost in valid_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")
    
            #test
            test_costs  =   []
            if iteration == 0:
                for i in range(len(test_X.get_value(borrow=True)) / batch_size):
                    x = test_X.get_value()[i * batch_size : (i+1) * batch_size]
                    cost = gsn_f_cost_init(x)
                    test_costs.append([cost])
            else:
                for i in range(len(test_X.get_value(borrow=True)) / batch_size):
                    xs = [test_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                    xs, _ = fix_input_size(xs)
                    costs = gsn_f_cost(*xs)
                    test_costs.append(costs)
                
            test_costs = numpy.mean(test_costs, 0) 
            print 'Test  : ', [trunc(cost) for cost in test_costs], '\t',
            with open(logfile,'a') as f:
                f.write("Test : {0!s}\t".format([trunc(cost) for cost in test_costs]))
            with open(test_convergence,'a') as f:
                for cost in test_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")
                
            #check for early stopping
            cost = numpy.sum(train_costs)
            if cost < best_cost*state.early_stop_threshold:
                patience = 0
                best_cost = cost
            else:
                patience += 1
    
            if counter >= n_epoch or patience >= state.early_stop_length:
                STOP = True
                save_params('gsn', counter, gsn_params, iteration)
                print "next learning rate should be", learning_rate.get_value() * annealing
    
            timing = time.time() - t
            times.append(timing)
    
            print 'time : ', trunc(timing),
            
            print 'remaining: ', trunc((n_epoch - counter) * numpy.mean(times) / 60 / 60), 'hrs'
    
            with open(logfile,'a') as f:
                f.write("MeanVisB : {0!s}\t".format(trunc(bias_list[0].get_value().mean())))
            
            with open(logfile,'a') as f:
                f.write("W : {0!s}\t".format(str([trunc(abs(w.get_value(borrow=True)).mean()) for w in weights_list])))
                
            with open(logfile,'a') as f:
                f.write("Time : {0!s} seconds\n".format(trunc(timing)))
    
            if (counter % state.save_frequency) == 0 or STOP is True:
                # Checking reconstruction
                # grab 100 numbers in the sequence from the test set
                nums = test_X.get_value()[range(100)]
                noisy_nums = f_noise(test_X.get_value()[range(100)])
                
                reconstructed_prediction = []
                reconstructed = []
                for i in range(100):
                    if i < len(Xs)-1:
                        xs = [noisy_nums[i-x] for x in range(i+1)]
                        xs.reverse()
                        # add filler to the remaining xs's (since we haven't seen enough in the sequence yet)
                        while len(xs) < len(Xs):
                            xs.append(noisy_nums[i])
                        _outs = f_recon(*xs)
                        predictions = _outs[:len(Xs)]
                        reconstructions = _outs[len(Xs):]
                        prediction = predictions[i]
                        reconstruction = reconstructions[i]
                    else:
                        xs = [noisy_nums[i-x] for x in range(len(Xs))]
                        xs.reverse()
                        _outs = f_recon(*xs)
                        predictions = _outs[:len(Xs)]
                        reconstructions = _outs[len(Xs):]
                        prediction = predictions[-1]
                        reconstruction = reconstructions[-1]
                    reconstructed_prediction.append(prediction)
                    reconstructed.append(reconstruction)
                # Concatenate stuff
                stacked = numpy.vstack([numpy.vstack([nums[i*10 : (i+1)*10], noisy_nums[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10], reconstructed_prediction[i*10 : (i+1)*10]]) for i in range(10)])
            
                number_reconstruction   =   PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,40)))
                #epoch_number    =   reduce(lambda x,y : x + y, ['_'] * (4-len(str(counter)))) + str(counter)
                number_reconstruction.save(outdir+'gsn_number_reconstruction_iteration_'+str(iteration)+'_epoch_'+str(counter)+'.png')
        
                #sample_numbers(counter, 'seven')
                plot_samples(counter, iteration)
        
                #save gsn_params
                save_params('gsn', counter, gsn_params, iteration)
         
            # ANNEAL!
            new_lr = learning_rate.get_value() * annealing
            learning_rate.set_value(new_lr)

        
        # 10k samples
        print 'Generating 10,000 samples'
        samples, _  =   sample_some_numbers(N=10000)
        f_samples   =   outdir+'samples.npy'
        numpy.save(f_samples, samples)
        print 'saved digits'
      
            
            
    #######################
    # REGRESSION TRAINING #
    #######################        
    def train_regression(iteration, train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
        print '-------------------------------------------'
        print 'TRAINING RECURRENT REGRESSION FOR ITERATION',iteration
        with open(logfile,'a') as f:
            f.write("\n\n--------------------------\nTRAINING RECURRENT REGRESSION FOR ITERATION {0!s}\n".format(iteration))
        
        # TRAINING
        n_epoch     =   state.n_epoch
        batch_size  =   state.batch_size
        STOP        =   False
        counter     =   0
        best_cost = float('inf')
        patience = 0
        if iteration == 0:
            regression_learning_rate.set_value(cast32(state.learning_rate))  # learning rate
        times = []
            
        print 'learning rate:',regression_learning_rate.get_value()
        
        print 'train X size:',str(train_X.shape.eval())
        print 'valid X size:',str(valid_X.shape.eval())
        print 'test X size:',str(test_X.shape.eval())
    
        if state.test_model:
            # If testing, do not train and go directly to generating samples, parzen window estimation, and inpainting
            print 'Testing : skip training'
            STOP    =   True

        while not STOP:
            counter += 1
            t = time.time()
            print counter,'\t',
            with open(logfile,'a') as f:
                f.write("{0!s}\t".format(counter))
                
            #shuffle the data
            data.sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, dataset, rng)
    
            #train
            train_costs = []
            for i in range(len(train_X.get_value(borrow=True)) / batch_size):
                xs = [train_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                xs, _ = fix_input_size(xs)
                costs = regression_f_learn(*xs)
                train_costs.append(costs)
                
            train_costs = numpy.mean(train_costs, 0) 
            print 'rTrain : ',[trunc(cost) for cost in train_costs], '\t',
            with open(logfile,'a') as f:
                f.write("rTrain : {0!s}\t".format([trunc(cost) for cost in train_costs]))
            with open(regression_train_convergence,'a') as f:
                for cost in train_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")
    
    
            #valid
            valid_costs  =  []
            for i in range(len(valid_X.get_value(borrow=True)) / batch_size):
                xs = [valid_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                xs, _ = fix_input_size(xs)
                costs = regression_f_cost(*xs)
                valid_costs.append(costs)
                    
            valid_costs = numpy.mean(valid_costs, 0)
            print 'rValid : ', [trunc(cost) for cost in valid_costs], '\t',
            with open(logfile,'a') as f:
                f.write("rValid : {0!s}\t".format([trunc(cost) for cost in valid_costs]))
            with open(regression_valid_convergence,'a') as f:
                for cost in valid_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")

    
            #test
            test_costs  =   []
            for i in range(len(test_X.get_value(borrow=True)) / batch_size):
                xs = [test_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                xs, _ = fix_input_size(xs)
                costs = regression_f_cost(*xs)
                test_costs.append(costs)
                
            test_costs = numpy.mean(test_costs, 0)
            print 'rTest  : ', [trunc(cost) for cost in test_costs], '\t',
            with open(logfile,'a') as f:
                f.write("rTest : {0!s}\t".format([trunc(cost) for cost in test_costs]))
            with open(regression_test_convergence,'a') as f:
                for cost in test_costs:
                    f.write("{0!s},".format(cost))
                f.write("\n")
    
            #check for early stopping
            cost = numpy.sum(train_costs)
            if cost < best_cost*state.early_stop_threshold:
                patience = 0
                best_cost = cost
            else:
                patience += 1
                
            if counter >= n_epoch or patience >= state.early_stop_length:
                STOP = True
                save_params('regression', counter, regression_params, iteration)
                print "next learning rate should be",regression_learning_rate.get_value() * annealing
    
            timing = time.time() - t
            times.append(timing)
    
            print 'time : ', trunc(timing),
            
            print 'remaining: ', trunc((n_epoch - counter) * numpy.mean(times) / 60 / 60), 'hrs'
                
            with open(logfile,'a') as f:
                f.write("Time : {0!s} seconds\n".format(trunc(timing)))
    
            if (counter % state.save_frequency) == 0 or STOP is True: 
                # Checking reconstruction
                # grab 100 numbers in the sequence from the test set
                nums = test_X.get_value()[range(100)]
                noisy_nums = f_noise(test_X.get_value()[range(100)])
                
                reconstructed_prediction = []
                reconstructed = []
                for i in range(100):
                    if i < len(Xs)-1:
                        xs = [noisy_nums[i-x] for x in range(i+1)]
                        xs.reverse()
                        # add filler to the remaining xs's (since we haven't seen enough in the sequence yet)
                        while len(xs) < len(Xs):
                            xs.append(noisy_nums[i])
                        _outs = f_recon(*xs)
                        predictions = _outs[:len(Xs)]
                        reconstructions = _outs[len(Xs):]
                        prediction = predictions[i]
                        reconstruction = reconstructions[i]
                    else:
                        xs = [noisy_nums[i-x] for x in range(len(Xs))]
                        xs.reverse()
                        _outs = f_recon(*xs)
                        predictions = _outs[:len(Xs)]
                        reconstructions = _outs[len(Xs):]
                        prediction = predictions[-1]
                        reconstruction = reconstructions[-1]
                    reconstructed_prediction.append(prediction)
                    reconstructed.append(reconstruction)
                # Concatenate stuff
                stacked = numpy.vstack([numpy.vstack([nums[i*10 : (i+1)*10], noisy_nums[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10], reconstructed_prediction[i*10 : (i+1)*10]]) for i in range(10)])
            
                number_reconstruction   =   PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,40)))
                #epoch_number    =   reduce(lambda x,y : x + y, ['_'] * (4-len(str(counter)))) + str(counter)
                number_reconstruction.save(outdir+'regression_number_reconstruction_iteration_'+str(iteration)+'_epoch_'+str(counter)+'.png')
             
                #save gsn_params
                save_params('regression', counter, regression_params, iteration)
         
            # ANNEAL!
            new_r_lr = regression_learning_rate.get_value() * annealing
            regression_learning_rate.set_value(new_r_lr)
            
            
            
            
    #####################
    # STORY 1 ALGORITHM #
    #####################
    
    for iteration in range(state.max_iterations):
        train_GSN(iteration, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
        train_regression(iteration, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
        
          
        
        
