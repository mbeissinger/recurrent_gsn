"""
.. module: generative_stochastic_network

This module gives an implementation of the Generative Stochastic Network model.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

This class's main() method by default produces the model trained on MNIST discussed in the paper:
'Deep Generative Stochastic Networks Trainable by Backprop'
Yoshua Bengio, Eric Thibodeau-Laufer
http://arxiv.org/abs/1306.1091

Scheduled noise is added as discussed in the paper:
'Scheduled denoising autoencoders'
Krzysztof J. Geras, Charles Sutton
http://arxiv.org/abs/1406.3269

TODO:
Multimodal transition operator (using NADE) discussed in:
'Multimodal Transitions for Generative Stochastic Networks'
Sherjil Ozair, Li Yao, Yoshua Bengio
http://arxiv.org/abs/1312.5578
"""

# standard libraries
# third-party libraries
import numpy

# internal references
# from utils.utils import cast32, sharedX, salt_and_pepper
# from utils.utils import add_gaussian_noise as add_gaussian
import utils.data_tools as data

# Default values to use for some GSN parameters. These defaults are used to produce the MNIST results given in the comments top of file.
_defaults = {  # gsn parameters
    "layers": 3,  # number of hidden layers to use
    "walkbacks": 5,
# number of walkbacks (generally 2*layers) - need enough to have info from top layer propagate to visible layer
    "hidden_size": 1000,  # number of hidden units in each layer
    "visible_activation": 'sigmoid',  # activation for visible layer - should be appropriate for input data type.
    "hidden_activation": 'tanh',  # activation for hidden layers
    "input_sampling": True,  # whether to sample at each walkback step - makes it like Gibbs sampling.
    # train param
    "cost_function": 'binary_crossentropy',
# the cost function to use during training - should be appropriate for input data type.
    # noise parameters
    "noise_decay": 'exponential',  # noise schedule algorithm
    "noise_annealing": 1.0,  # no noise schedule by default
    "add_noise": True,  # whether to add noise throughout the network's hidden layers
    "noiseless_h1": True,  # whether to keep the first hidden layer uncorrupted
    "hidden_add_noise_sigma": 2,  # sigma value for adding the gaussian hidden layer noise
    "input_salt_and_pepper": 0.4,  # the salt and pepper value for inputs corruption
    # data parameters
    "output_path": 'outputs/gsn/',  # base directory to output various files
    "is_image": True,  # whether the input should be treated as an image
    "vis_init": False}


class GSN_OLD:
    """
    Class for creating a new Generative Stochastic Network (GSN)
    """

    def __init__(self, train_X, valid_X, test_X, state, outdir_base='./', logger=None):
        pass
###############################################
# COMPUTATIONAL GRAPH HELPER METHODS FOR GSN #
###############################################
    @staticmethod
    def update_layers(hiddens,
                      weights_list,
                      bias_list,
                      p_X_chain,
                      add_noise=_defaults["add_noise"],
                      noiseless_h1=_defaults["noiseless_h1"],
                      hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                      input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                      input_sampling=_defaults["input_sampling"],
                      visible_activation=_defaults["visible_activation"],
                      hidden_activation=_defaults["hidden_activation"],
                      logger=None):
        # One update over the odd layers + one update over the even layers
        # update the odd layers
        GSN.update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma,
                              input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        # update the even layers
        GSN.update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                               hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                               hidden_activation, logger)

    @staticmethod
    def update_layers_scan_step(hiddens_t,
                                weights_list,
                                bias_list,
                                add_noise=_defaults["add_noise"],
                                noiseless_h1=_defaults["noiseless_h1"],
                                hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                                input_sampling=_defaults["input_sampling"],
                                MRG=_defaults["MRG"],
                                visible_activation=_defaults["visible_activation"],
                                hidden_activation=_defaults["hidden_activation"],
                                logger=None):
        p_X_chain = []
        # One update over the odd layers + one update over the even layers
        # update the odd layers
        GSN.update_odd_layers(hiddens_t, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma,
                              input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)
        # update the even layers
        GSN.update_even_layers(hiddens_t, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                               hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                               hidden_activation, logger)
        # return the generated sample, the sampled next input, and hiddens
        return p_X_chain[0], hiddens_t

    @staticmethod
    def update_layers_reverse(hiddens,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise=_defaults["add_noise"],
                              noiseless_h1=_defaults["noiseless_h1"],
                              hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                              input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                              input_sampling=_defaults["input_sampling"],
                              MRG=_defaults["MRG"],
                              visible_activation=_defaults["visible_activation"],
                              hidden_activation=_defaults["hidden_activation"],
                              logger=None):
        # One update over the even layers + one update over the odd layers
        # update the even layers
        GSN.update_even_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                               hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                               hidden_activation, logger)
        # update the odd layers
        GSN.update_odd_layers(hiddens, weights_list, bias_list, add_noise, noiseless_h1, hidden_add_noise_sigma,
                              input_salt_and_pepper, input_sampling, MRG, visible_activation, hidden_activation, logger)

    # Odd layer update function
    # just a loop over the odd layers
    @staticmethod
    def update_odd_layers(hiddens,
                          weights_list,
                          bias_list,
                          add_noise=_defaults["add_noise"],
                          noiseless_h1=_defaults["noiseless_h1"],
                          hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                          input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                          input_sampling=_defaults["input_sampling"],
                          MRG=_defaults["MRG"],
                          visible_activation=_defaults["visible_activation"],
                          hidden_activation=_defaults["hidden_activation"],
                          logger=None):
        # Loop over the odd layers
        for i in range(1, len(hiddens), 2):
            GSN.simple_update_layer(hiddens, weights_list, bias_list, None, i, add_noise, noiseless_h1,
                                    hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG,
                                    visible_activation, hidden_activation, logger)

    # Even layer update
    # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
    @staticmethod
    def update_even_layers(hiddens,
                           weights_list,
                           bias_list,
                           p_X_chain,
                           add_noise=_defaults["add_noise"],
                           noiseless_h1=_defaults["noiseless_h1"],
                           hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                           input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                           input_sampling=_defaults["input_sampling"],
                           MRG=_defaults["MRG"],
                           visible_activation=_defaults["visible_activation"],
                           hidden_activation=_defaults["hidden_activation"],
                           logger=None):
        # Loop over even layers
        for i in range(0, len(hiddens), 2):
            GSN.simple_update_layer(hiddens, weights_list, bias_list, p_X_chain, i, add_noise, noiseless_h1,
                                    hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG,
                                    visible_activation, hidden_activation, logger)

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
                            add_noise=_defaults["add_noise"],
                            noiseless_h1=_defaults["noiseless_h1"],
                            hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                            input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                            input_sampling=_defaults["input_sampling"],
                            MRG=_defaults["MRG"],
                            visible_activation=_defaults["visible_activation"],
                            hidden_activation=_defaults["hidden_activation"],
                            logger=None):
        # Compute the dot product, whatever layer
        # If the visible layer X
        if i == 0:
            hiddens[i] = T.dot(hiddens[i + 1], weights_list[i].T) + bias_list[i]
        # If the top layer
        elif i == len(hiddens) - 1:
            hiddens[i] = T.dot(hiddens[i - 1], weights_list[i - 1]) + bias_list[i]
        # Otherwise in-between layers
        else:
            # next layer        :   hiddens[i+1], assigned weights : W_i
            # previous layer    :   hiddens[i-1], assigned weights : W_(i-1)
            hiddens[i] = T.dot(hiddens[i + 1], weights_list[i].T) + T.dot(hiddens[i - 1], weights_list[i - 1]) + \
                         bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i == 1 and noiseless_h1:
            add_noise = False

        # pre activation noise
        if i != 0 and add_noise:
            hiddens[i] = add_gaussian(hiddens[i], std=hidden_add_noise_sigma, MRG=MRG)

        # ACTIVATION!
        if i == 0:
            hiddens[i] = visible_activation(hiddens[i])
        else:
            hiddens[i] = hidden_activation(hiddens[i])

        # post activation noise
        # why is there post activation noise? Because there is already pre-activation noise, this just doubles the amount of noise between each activation of the hiddens.
        if i != 0 and add_noise:
            hiddens[i] = add_gaussian(hiddens[i], std=hidden_add_noise_sigma, MRG=MRG)

        # build the reconstruction chain if updating the visible layer X
        if i == 0:
            # if input layer -> append p(X|H...)
            p_X_chain.append(hiddens[i])

            # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
            if input_sampling:
                sampled = MRG.binomial(p=hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
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
                  add_noise=_defaults["add_noise"],
                  noiseless_h1=_defaults["noiseless_h1"],
                  hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                  input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                  input_sampling=_defaults["input_sampling"],
                  MRG=_defaults["MRG"],
                  visible_activation=_defaults["visible_activation"],
                  hidden_activation=_defaults["hidden_activation"],
                  walkbacks=_defaults["walkbacks"]):
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
        for i in range(walkbacks):
            GSN.update_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                              hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                              hidden_activation)

        return p_X_chain, hiddens

    @staticmethod
    def build_gsn_given_hiddens(X,
                                hiddens,
                                weights_list,
                                bias_list,
                                add_noise=_defaults["add_noise"],
                                noiseless_h1=_defaults["noiseless_h1"],
                                hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                                input_sampling=_defaults["input_sampling"],
                                MRG=_defaults["MRG"],
                                visible_activation=_defaults["visible_activation"],
                                hidden_activation=_defaults["hidden_activation"],
                                walkbacks=_defaults["walkbacks"],
                                cost_function=_defaults["cost_function"]):

        p_X_chain = []
        for i in range(walkbacks):
            GSN.update_layers_reverse(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                                      hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG,
                                      visible_activation, hidden_activation)

        # x_sample = p_X_chain[-1]

        costs = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1]  # for logging to show progress
        cost = numpy.sum(costs)

        mse = T.mean(T.sqr(p_X_chain[-1] - X), axis=0)
        error = T.mean(mse)

        return p_X_chain, hiddens, cost, show_cost, error

    @staticmethod
    def build_gsn_scan(X,
                       weights_list,
                       bias_list,
                       add_noise=_defaults["add_noise"],
                       noiseless_h1=_defaults["noiseless_h1"],
                       hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                       input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                       input_sampling=_defaults["input_sampling"],
                       MRG=_defaults["MRG"],
                       visible_activation=_defaults["visible_activation"],
                       hidden_activation=_defaults["hidden_activation"],
                       walkbacks=_defaults["walkbacks"],
                       cost_function=_defaults["cost_function"]):

        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(X, input_salt_and_pepper, MRG)
        else:
            X_init = X
        # init hiddens with zeros
        hiddens_0 = [X_init]
        for w in weights_list:
            hiddens_0.append(T.zeros_like(T.dot(hiddens_0[-1], w)))

        p_X_chain = []
        for i in range(walkbacks):
            GSN.update_layers(hiddens_0, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                              hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                              hidden_activation)

        x_sample = p_X_chain[-1]

        costs = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1]  # for logging to show progress
        cost = numpy.sum(costs)

        return x_sample, cost, show_cost  # , updates

    @staticmethod
    def build_gsn_pxh(hiddens,
                      weights_list,
                      bias_list,
                      add_noise=_defaults["add_noise"],
                      noiseless_h1=_defaults["noiseless_h1"],
                      hidden_add_noise_sigma=_defaults["hidden_add_noise_sigma"],
                      input_salt_and_pepper=_defaults["input_salt_and_pepper"],
                      input_sampling=_defaults["input_sampling"],
                      MRG=_defaults["MRG"],
                      visible_activation=_defaults["visible_activation"],
                      hidden_activation=_defaults["hidden_activation"],
                      walkbacks=_defaults["walkbacks"]):

        p_X_chain = []
        for i in range(walkbacks):
            GSN.update_layers(hiddens, weights_list, bias_list, p_X_chain, add_noise, noiseless_h1,
                              hidden_add_noise_sigma, input_salt_and_pepper, input_sampling, MRG, visible_activation,
                              hidden_activation)

        x_sample = p_X_chain[-1]

        return x_sample
