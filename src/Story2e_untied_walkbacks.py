import numpy, os, sys, cPickle
import numpy.random as rng
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time
import argparse
import data_tools as data

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

def sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
    #shuffle the datasets
    train_indices = range(len(train_Y.get_value(borrow=True)))
    rng.shuffle(train_indices)
    valid_indices = range(len(valid_Y.get_value(borrow=True)))
    rng.shuffle(valid_indices)
    test_indices = range(len(test_Y.get_value(borrow=True)))
    rng.shuffle(test_indices)
    
    train_X.set_value(train_X.get_value(borrow=True)[train_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_indices])
    
    # Find the order of MNIST data going from 0-9 repeating
    train_ordered_indices = data.create_series(train_Y.get_value(borrow=True), 10)
    valid_ordered_indices = data.create_series(valid_Y.get_value(borrow=True), 10)
    test_ordered_indices = data.create_series(test_Y.get_value(borrow=True), 10)
    
    # Put the data sets in order
    train_X.set_value(train_X.get_value(borrow=True)[train_ordered_indices])
    train_Y.set_value(train_Y.get_value(borrow=True)[train_ordered_indices])
    
    valid_X.set_value(valid_X.get_value(borrow=True)[valid_ordered_indices])
    valid_Y.set_value(valid_Y.get_value(borrow=True)[valid_ordered_indices])
    
    test_X.set_value(test_X.get_value(borrow=True)[test_ordered_indices])
    test_Y.set_value(test_Y.get_value(borrow=True)[test_ordered_indices])



def experiment(state, outdir='./'):
    logfile = outdir+"log.txt"
    with open(logfile,'w') as f:
        f.write("MODEL 2\n\n")
    print
    print "----------MODEL 2--------------"
    print
    if state.test_model and 'config' in os.listdir('.'):
        print 'Loading local config file'
        config_file =   open('config', 'r')
        config      =   config_file.readlines()
        try:
            config_vals =   config[0].split('(')[1:][0].split(')')[:-1][0].split(', ')
        except:
            config_vals =   config[0][3:-1].replace(': ','=').replace("'","").split(', ')
            config_vals =   filter(lambda x:not 'jobman' in x and not '/' in x and not ':' in x and not 'experiment' in x, config_vals)
        
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
        with open(outdir+'config', 'w') as f:
            f.write(str(state))


    print state
    # Load the data, train = train+valid, and shuffle train
    # Targets are not used (will be misaligned after shuffling train
    if state.dataset == 'MNIST':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))
    elif state.dataset == 'MNIST_binary':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.load_mnist_binary(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))
    else:
        raise AssertionError("dataset not recognized.")
    
    rng.seed(1) #seed the numpy random generator  
    
    print 'Sequencing MNIST data...'
    print 'train set size:',len(train_Y)
    print 'valid set size:',len(valid_Y)
    print 'test set size:',len(test_Y)
    
    train_X = theano.shared(train_X)
    train_Y = theano.shared(train_Y)
    valid_X = theano.shared(valid_X)
    valid_Y = theano.shared(valid_Y) 
    test_X = theano.shared(test_X)
    test_Y = theano.shared(test_Y) 
   
    sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
    
    print 'train set size:',len(train_Y.eval())
    print 'valid set size:',len(valid_Y.eval())
    print 'test set size:',len(test_Y.eval())
    print 'Sequencing done.'
    
    N_input =   train_X.eval().shape[1]
    root_N_input = numpy.sqrt(N_input)  
    
    
    
    # Network and training specifications
    layers          =   state.layers # number hidden layers
    walkbacks       =   state.walkbacks # number of walkbacks 
    layer_sizes     =   [N_input] + [state.hidden_size] * layers # layer sizes, from h0 to hK (h0 is the visible layer)
    learning_rate   =   theano.shared(cast32(state.learning_rate))  # learning rate
    annealing       =   cast32(state.annealing) # exponential annealing coefficient
    momentum        =   theano.shared(cast32(state.momentum)) # momentum term 

    # PARAMETERS : weights list and bias list.
    # initialize a list of weights and biases based on layer_sizes
    weights_list    =   [get_shared_weights(layer_sizes[i], layer_sizes[i+1], name="W_{0!s}_{1!s}".format(i,i+1)) for i in range(layers)] # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
    recurrent_weights_list    =   [get_shared_weights(layer_sizes[i+1], layer_sizes[i], name="V_{0!s}_{1!s}".format(i+1,i)) for i in range(layers)] # initialize each layer to uniform sample from sqrt(6. / (n_in + n_out))
    bias_list       =   [get_shared_bias(layer_sizes[i], name='b_'+str(i)) for i in range(layers + 1)] # initialize each layer to 0's.
    
    # Theano variables and RNG
    MRG = RNG_MRG.MRG_RandomStreams(1)
    X = T.fmatrix('X')
    Xs = [T.fmatrix(name="X_initial") if i==0 else T.fmatrix(name="X_"+str(i+1)) for i in range(walkbacks+1)]
    hiddens_input = [X] + [T.fmatrix(name="h_"+str(i+1)) for i in range(layers)]
    hiddens_output = hiddens_input[:1] + hiddens_input[1:]
    

    if state.test_model:
        # Load the parameters of the last epoch
        # maybe if the path is given, load these specific attributes 
        param_files     =   filter(lambda x:'params' in x, os.listdir('.'))
        max_epoch_idx   =   numpy.argmax([int(x.split('_')[-1].split('.')[0]) for x in param_files])
        params_to_load  =   param_files[max_epoch_idx]
        PARAMS = cPickle.load(open(params_to_load,'r'))
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]
        
    if state.continue_training:
        # Load the parameters of the last GSN
        params_to_load = 'gsn_params.pkl'
        PARAMS = cPickle.load(open(params_to_load,'r'))
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]

 
    ''' F PROP '''
    if state.act == 'sigmoid':
        print 'Using sigmoid activation'
        hidden_activation = T.nnet.sigmoid
    elif state.act == 'rectifier':
        print 'Using rectifier activation'
        hidden_activation = lambda x : T.maximum(cast32(0), x)
    elif state.act == 'tanh':
        hidden_activation = lambda x : T.tanh(x)
        
    visible_activation = T.nnet.sigmoid 
  
        
    def update_layers(hiddens, p_X_chain, Xs, sequence_idx, direction, noisy = True):
        print 'odd layer updates'
        update_odd_layers(hiddens, direction, noisy)
        print 'even layer updates'
        update_even_layers(hiddens, p_X_chain, Xs, sequence_idx, direction, noisy)
        print 'done full update.'
        print
        
    # Odd layer update function
    # just a loop over the odd layers
    def update_odd_layers(hiddens, direction, noisy):
        for i in range(1, len(hiddens), 2):
            print 'updating layer',i
            simple_update_layer(hiddens, None, None, None, i, direction = direction, add_noise = noisy)
    
    # Even layer update
    # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
    def update_even_layers(hiddens, p_X_chain, Xs, sequence_idx, direction, noisy):
        for i in range(0, len(hiddens), 2):
            print 'updating layer',i
            simple_update_layer(hiddens, p_X_chain, Xs, sequence_idx, i, direction = direction, add_noise = noisy)
    
    # The layer update function
    # hiddens   :   list containing the symbolic theano variables [visible, hidden1, hidden2, ...]
    #               layer_update will modify this list inplace
    # p_X_chain :   list containing the successive p(X|...) at each update
    #               update_layer will append to this list
    # add_noise     : pre and post activation gaussian noise
    
    def simple_update_layer(hiddens, p_X_chain, Xs, sequence_idx, i, direction = "forward", add_noise=True):             
        # Compute the dot product, whatever layer
        # If the visible layer X
        if i == 0:
            if direction == "forward":
                print 'using '+str(recurrent_weights_list[i])
                hiddens[i] = (T.dot(hiddens[i+1], recurrent_weights_list[i]) + bias_list[i])
            else:
                print 'using', str(weights_list[i])+'.T'
                hiddens[i] = (T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i])
        # If the top layer
        elif i == len(hiddens)-1:
            print 'using',weights_list[i-1]
            hiddens[i]  =   T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
        # Otherwise in-between layers
        else:
            if direction == "forward":
                print "using {0!s} and {1!s}".format(weights_list[i-1],recurrent_weights_list[i])
                hiddens[i]  =   T.dot(hiddens[i+1], recurrent_weights_list[i]) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
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
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
       
        # ACTIVATION!
        if i == 0:
            print 'Sigmoid units activation for visible layer X'
            hiddens[i]  =   visible_activation(hiddens[i])
        else:
            print 'Hidden units {} activation for layer'.format(state.act), i
            hiddens[i]  =   hidden_activation(hiddens[i])
    
        # post activation noise            
        if i != 0 and add_noise:
            print 'Adding post-activation gaussian noise for layer', i
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
    
        # build the reconstruction chain if updating the visible layer X
        if i == 0:
            # if input layer -> append p(X|...)
            p_X_chain.append(hiddens[i]) #what the predicted next input should be
            
            if direction == "forward":
                if sequence_idx+1 < len(Xs):
                    next = Xs[sequence_idx+1]
                    # sample from p(X|...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
                    if state.input_sampling:
                        print 'Sampling from input'
                        sampled     =   MRG.binomial(p = next, size=next.shape, dtype='float32')
                    else:
                        print '>>NO input sampling'
                        sampled     =   next
                    # add noise
                    sampled     =   salt_and_pepper(sampled, state.input_salt_and_pepper)
                    
                    # set input layer
                    hiddens[i]  =   sampled
            else:
                # sample from p(X|...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
                if state.input_sampling:
                    print 'Sampling from input'
                    sampled     =   MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
                else:
                    print '>>NO input sampling'
                    sampled     =   hiddens[i]
                # add noise
                sampled     =   salt_and_pepper(sampled, state.input_salt_and_pepper)
                
                # set input layer
                hiddens[i]  =   sampled
            
    
    ''' Corrupt X '''
    predicted_X_chain    = []
    hiddens_output[0] = salt_and_pepper(hiddens_output[0], state.input_salt_and_pepper)
    #hiddens[0] = salt_and_pepper(Xs[0], state.input_salt_and_pepper)
    # The layer update scheme
    print "Building the graph :", walkbacks,"updates"
    for i in range(walkbacks):
        print "Forward Prediction {!s}/{!s}".format(i+1,walkbacks)
        update_layers(hiddens_output, predicted_X_chain, Xs, i, direction="forward", noisy=True)
        
#     p_X_chain    = []
#     X_corrupt   = salt_and_pepper(Xs[0], state.input_salt_and_pepper)
#     hiddens     = [X_corrupt]
#     print "Hidden units initialization"
#     for w in weights_list:
#         # init with zeros
#         print "Init hidden units at zero before creating the graph"
#         print
#         hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
#     # The layer update scheme
#     print "Building the graph :", walkbacks,"updates"
#     for i in range(walkbacks):
#         print "Backward Prediction {!s}/{!s}".format(i+1,walkbacks)
#         update_layers(hiddens, p_X_chain, Xs, i, direction="back", noisy=True)

    # COST AND GRADIENTS    
    print
    print 'Cost w.r.t p(X|...) at every step in the graph'
    
    costs = [T.mean(T.nnet.binary_crossentropy(predicted_X_chain[i], Xs[i+1])) for i in range(len(predicted_X_chain))]
    pred = predicted_X_chain[0]
    show_COSTs = [costs[0]] + [costs[-1]]
    COST = T.sum(costs)
    
    params      =   weights_list + recurrent_weights_list + bias_list
    print "params:",params
    
    print "creating functions..."    
    gradient        =   T.grad(COST, params)
                
    gradient_buffer =   [theano.shared(numpy.zeros(param.get_value().shape, dtype='float32')) for param in params]
    
    m_gradient      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
    param_updates   =   [(param, param - learning_rate * mg) for (param, mg) in zip(params, m_gradient)]
    gradient_buffer_updates = zip(gradient_buffer, m_gradient)
        
    updates         =   OrderedDict(param_updates + gradient_buffer_updates)
    
    
    #odd layer h's not used from input -> calculated directly from even layers (starting with h_0) since the odd layers are updated first.
    f_cost          =   theano.function(inputs = hiddens_input + Xs, outputs = hiddens_output + show_COSTs, on_unused_input='warn')

    f_learn         =   theano.function(inputs  = hiddens_input + Xs,
                                        updates = updates,
                                        outputs = hiddens_output + show_COSTs,
                                        on_unused_input='warn')
    

    
    print "functions done."
    print
    
    #############
    # Denoise some numbers  :   show number, noisy number, reconstructed number
    #############
    import random as R
    R.seed(1)
    # Grab 100 random indices from test_X
    random_idx      =   numpy.array(R.sample(range(len(test_X.get_value())), 100))
    numbers         =   test_X.get_value()[random_idx]
    
    f_noise         =   theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))
    noisy_numbers   =   f_noise(test_X.get_value()[random_idx])
    #noisy_numbers   =   salt_and_pepper(numbers, state.input_salt_and_pepper)

    # Recompile the graph without noise for reconstruction function
    X_recon = T.fmatrix()
    hiddens_R     = [X_recon]
    p_X_chain_R   = []

    for w in weights_list:
        # init with zeros
        hiddens_R.append(T.zeros_like(T.dot(hiddens_R[-1], w)))

    # The layer update scheme
    print "Creating graph for noisy reconstruction function at checkpoints during training."
    noiseflag = False
    for i in range(layers):
        print "Walkback {!s}/{!s}".format(i+1,layers)
        update_layers(hiddens_R, p_X_chain_R, [X_recon], i, direction="forward", noisy=noiseflag)

    f_recon = theano.function(inputs = [X_recon], outputs = [p_X_chain_R[0] ,p_X_chain_R[-1]]) 


    ############
    # Sampling #
    ############
    
    # the input to the sampling function
    
    network_state_input     =   [X] + [T.fmatrix() for i in range(layers)]
   
    # "Output" state of the network (noisy)
    # initialized with input, then we apply updates
    
    network_state_output    =   [X] + network_state_input[1:]

    visible_pX_chain        =   []

    # ONE update
    print "Performing one walkback in network state sampling."
    update_layers(network_state_output, visible_pX_chain, [X], 0, direction="forward", noisy=True)

    if layers == 1: 
        f_sample_simple = theano.function(inputs = [X], outputs = visible_pX_chain[-1])
    
    
    # WHY IS THERE A WARNING????
    # because the first odd layers are not used -> directly computed FROM THE EVEN layers
    # unused input = warn
    f_sample2   =   theano.function(inputs = network_state_input, outputs = network_state_output + visible_pX_chain, on_unused_input='warn')
    #f_sample2   =   theano.function(inputs = network_state_input, outputs = network_state_output + visible_pX_chain)

    def sample_some_numbers_single_layer():
        x0    =   test_X.get_value()[:1]
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
        init_vis        =   test_X.get_value()[:1]

        noisy_init_vis  =   f_noise(init_vis)

        network_state   =   [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [noisy_init_vis]

        for i in range(N-1):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

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
            V, H0 = sample_some_numbers()
        img_samples =   PIL.Image.fromarray(tile_raster_images(V, (root_N_input,root_N_input), (20,20)))
        
        fname       =   outdir+'samples_iteration_'+str(iteration)+'_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        print 'Took ' + str(time.time() - to_sample) + ' to sample 400 numbers'
   
    ##############
    # Inpainting #
    ##############
    def inpainting(digit):
        # The network's initial state

        # NOISE INIT
        init_vis    =   cast32(numpy.random.uniform(size=digit.shape))

        #noisy_init_vis  =   f_noise(init_vis)
        #noisy_init_vis  =   cast32(numpy.random.uniform(size=init_vis.shape))

        # INDEXES FOR VISIBLE AND NOISY PART
        noise_idx = (numpy.arange(N_input) % root_N_input < (root_N_input/2))
        fixed_idx = (numpy.arange(N_input) % root_N_input > (root_N_input/2))
        # function to re-init the visible to the same noise

        # FUNCTION TO RESET HALF VISIBLE TO DIGIT
        def reset_vis(V):
            V[0][fixed_idx] =   digit[0][fixed_idx]
            return V
        
        # INIT DIGIT : NOISE and RESET HALF TO DIGIT
        init_vis = reset_vis(init_vis)

        network_state   =   [[init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [init_vis]

        for i in range(49):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])


            # reset half the digit
            net_state_out[0] = reset_vis(net_state_out[0])
            vis_pX_chain[0]  = reset_vis(vis_pX_chain[0])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)

    def save_params(name, n, params, iteration):
        print 'saving parameters...'
        save_path = outdir+name+'_params_iteration_'+str(iteration)+'_epoch_'+str(n)+'.pkl'
        f = open(save_path, 'wb')
        try:
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close() 

    def fix_input_size(hiddens, xs):
        sizes = [x.shape[0] for x in xs]
        min_size = numpy.min(sizes)
        xs = [x[:min_size] for x in xs]
        hiddens = [xs[0] if i==0 else hiddens[i][:min_size] for i in range(len(hiddens))]
        return hiddens, xs

    ################
    # GSN TRAINING #
    ################
    def train_recurrent_GSN(iteration, train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
        print '----------------------------------------'
        print 'TRAINING GSN FOR ITERATION',iteration
        with open(logfile,'a') as f:
            f.write("--------------------------\nTRAINING GSN FOR ITERATION {0!s}".format(iteration))
        
        # TRAINING
        n_epoch     =   state.n_epoch
        batch_size  =   state.batch_size
        STOP        =   False
        counter     =   0
        if iteration == 0:
            learning_rate.set_value(cast32(state.learning_rate))  # learning rate
        times = []
            
        print 'learning rate:',learning_rate.get_value()
        
        print 'train X size:',str(train_X.shape.eval())
        print 'valid X size:',str(valid_X.shape.eval())
        print 'test X size:',str(test_X.shape.eval())
    
        train_costs =   []
        valid_costs =   []
        test_costs =   []
        train_costs_post =   []
        valid_costs_post =   []
        test_costs_post =   []
        
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
            sequence_mnist_data(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
            
            #train
            #init hiddens
            hiddens = [(T.zeros_like(train_X[:batch_size]).eval())]
            for i in range(len(weights_list)):
                # init with zeros
                hiddens.append(T.zeros_like(T.dot(hiddens[i], weights_list[i])).eval())
            train_cost = []
            train_cost_post = []
            for i in range(len(train_X.get_value(borrow=True)) / batch_size):
                xs = [train_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                hiddens, xs = fix_input_size(hiddens, xs)
                if i==500:
                    print "hiddens {0!s}:".format(i),
                    for h in hiddens:
                        print trunc(numpy.mean(h)), trunc(numpy.min(h)), trunc(numpy.max(h)), "|",
                _ins = hiddens + xs
                _outs = f_learn(*_ins)
                hiddens = _outs[:len(hiddens)]
                cost = _outs[-2]
                cost_post = _outs[-1]
                train_cost.append(cost)
                train_cost_post.append(cost_post)
                
            train_cost = numpy.mean(train_cost)
            train_costs.append(train_cost)
            train_cost_post = numpy.mean(train_cost_post)
            train_costs_post.append(train_cost_post)
            print 'Train : ',trunc(train_cost),trunc(train_cost_post), '\t',
            with open(logfile,'a') as f:
                f.write("Train : {0!s} {1!s}\t".format(trunc(train_cost),trunc(train_cost_post)))
    
    
            #valid
            #init hiddens
            hiddens[0] = (T.zeros_like(valid_X[:batch_size]).eval())
            for i in range(len(weights_list)):
                # init with zeros
                hiddens[i+1] = (T.zeros_like(T.dot(hiddens[i], weights_list[i])).eval())
            valid_cost = []
            valid_cost_post = []
            for i in range(len(valid_X.get_value(borrow=True)) / batch_size):
                xs = [valid_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                hiddens, xs = fix_input_size(hiddens, xs)
                _ins = hiddens + xs
                _outs = f_cost(*_ins)
                hiddens = _outs[:-2]
                cost = _outs[-2]
                cost_post = _outs[-1]
                valid_cost.append(cost)
                valid_cost_post.append(cost_post)

            valid_cost = numpy.mean(valid_cost)
            valid_costs.append(valid_cost)
            valid_cost_post = numpy.mean(valid_cost_post)
            valid_costs_post.append(valid_cost_post)
            print 'Valid : ', trunc(valid_cost),trunc(valid_cost_post), '\t',
            with open(logfile,'a') as f:
                f.write("Valid : {0!s} {1!s}\t".format(trunc(valid_cost),trunc(valid_cost_post)))
    
    
            #test
            #init hiddens
            hiddens[0] = (T.zeros_like(test_X[:batch_size]).eval())
            for i in range(len(weights_list)):
                # init with zeros
                hiddens[i+1] = (T.zeros_like(T.dot(hiddens[i], weights_list[i])).eval())
            test_cost = []
            test_cost_post = []
            for i in range(len(test_X.get_value(borrow=True)) / batch_size):
                xs = [test_X.get_value(borrow=True)[(i * batch_size) + sequence_idx : ((i+1) * batch_size) + sequence_idx] for sequence_idx in range(len(Xs))]
                hiddens, xs = fix_input_size(hiddens, xs)
                _ins = hiddens + xs
                _outs = f_cost(*_ins)
                hiddens = _outs[:-2]
                cost = _outs[-2]
                cost_post = _outs[-1]
                test_cost.append(cost)
                test_cost_post.append(cost_post)

            test_cost = numpy.mean(test_cost)
            test_costs.append(test_cost)
            test_cost_post = numpy.mean(test_cost_post)
            test_costs_post.append(test_cost_post)
            print 'Test  : ', trunc(test_cost),trunc(test_cost_post), '\t',
            with open(logfile,'a') as f:
                f.write("Test : {0!s} {1!s}\t".format(trunc(test_cost),trunc(test_cost_post)))
            
    
            if counter >= n_epoch:
                STOP = True
                save_params('gsn', counter, params, iteration)
    
            timing = time.time() - t
            times.append(timing)
    
            print 'time : ', trunc(timing),
            
            print 'remaining: ', trunc((n_epoch - counter) * numpy.mean(times) / 60 / 60), 'hrs',
    
            print 'B : ', [trunc(abs(b.get_value(borrow=True)).mean()) for b in bias_list],
            
            print 'W : ', [trunc(abs(w.get_value(borrow=True)).mean()) for w in weights_list],
            
            print 'V : ', [trunc(abs(v.get_value(borrow=True)).mean()) for v in recurrent_weights_list]
    
            if (counter % state.save_frequency) == 0:
                # Checking reconstruction
                reconstructed_prediction, reconstructed_prediction_end   =   f_recon(noisy_numbers) 
                # Concatenate stuff
                stacked = numpy.vstack([numpy.vstack([numbers[i*10 : (i+1)*10], noisy_numbers[i*10 : (i+1)*10], reconstructed_prediction[i*10 : (i+1)*10], reconstructed_prediction_end[i*10 : (i+1)*10]]) for i in range(10)])
            
                numbers_reconstruction   =   PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,40)))
                #epoch_number    =   reduce(lambda x,y : x + y, ['_'] * (4-len(str(counter)))) + str(counter)
                numbers_reconstruction.save(outdir+'gsn_number_reconstruction_iteration_'+str(iteration)+'_epoch_'+str(counter)+'.png')
        
                #sample_numbers(counter, 'seven')
                plot_samples(counter, iteration)
        
                #save params
                save_params('gsn', counter, params, iteration)
         
            # ANNEAL!
            new_lr = learning_rate.get_value() * annealing
            learning_rate.set_value(new_lr)
    
        # Save
        state.train_costs = train_costs
        state.valid_costs = valid_costs
        state.test_costs = test_costs
    
    
        # 10k samples
        print 'Generating 10,000 samples'
        samples, _  =   sample_some_numbers(N=10000)
        f_samples   =   outdir+'samples.npy'
        numpy.save(f_samples, samples)
        print 'saved digits'
    
    
        # parzen
#         print 'Evaluating parzen window'
#         import likelihood_estimation_parzen
#         likelihood_estimation_parzen.main(0.20,'mnist') 
    
        # Inpainting
        '''
        print 'Inpainting'
        test_X  =   test_X.get_value()
    
        numpy.random.seed(2)
        test_idx    =   numpy.arange(len(test_Y.get_value(borrow=True)))
    
        for Iter in range(10):
    
            numpy.random.shuffle(test_idx)
            test_X = test_X[test_idx]
            test_Y = test_Y[test_idx]
    
            digit_idx = [(test_Y==i).argmax() for i in range(10)]
            inpaint_list = []
    
            for idx in digit_idx:
                DIGIT = test_X[idx:idx+1]
                V_inpaint, H_inpaint = inpainting(DIGIT)
                inpaint_list.append(V_inpaint)
    
            INPAINTING  =   numpy.vstack(inpaint_list)
    
            plot_inpainting =   PIL.Image.fromarray(tile_raster_images(INPAINTING, (root_N_input,root_N_input), (10,50)))
    
            fname   =   'inpainting_'+str(Iter)+'_iteration_'+str(iteration)+'.png'
            #fname   =   os.path.join(state.model_path, fname)
    
            plot_inpainting.save(fname)
    '''        
            
            
            
    #####################
    # STORY 2 ALGORITHM #
    #####################
    for iter in range(state.max_iterations):
        train_recurrent_GSN(iter, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)        
        
        
        

    if __name__ == '__main__':
        import ipdb; ipdb.set_trace() 
        return 
