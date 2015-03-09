__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
# internal references
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights_gaussian, get_weights_uniform, get_bias
from opendeep.utils.activation import get_activation_function

log = logging.getLogger(__name__)

class FullyConnectedLayer(Model):
    """
    This is your generic input -> nonlinear(output) layer. No hidden representation.
    """
    default = {
        'activation': 'rectifier',  # type of activation function to use for output
        'weights_init': 'gaussian',  # either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0,  # mean for gaussian weights init
        'weights_std': 0.005,  # standard deviation for gaussian weights init
        'weights_interval': 'good',  # if the weights_init was 'uniform', how to initialize from uniform
        'bias_init': 0.0  # how to initialize the bias parameter
    }
    def __init__(self, inputs_hook, config=None, defaults=default, params_hook=None):
        # init Model to combine the defaults and config dictionaries.
        super(FullyConnectedLayer, self).__init__(config, defaults)
        # all configuration parameters are now in self.args

        # grab info from the inputs_hook
        shape = inputs_hook[0]
        self.input = inputs_hook[1]

        # parameters - make sure to deal with params_hook!
        if params_hook:
            assert len(params_hook) == 2
            W, b = params_hook
        else:
            # if we are initializing weights from a gaussian
            if self.args.get('weights_init').lower() == 'gaussian':
                mean = self.args.get('weights_mean')
                std  = self.args.get('weights_std')
                # if both mean and std are supplied
                if mean and std:
                    W = get_weights_gaussian(shape=shape, mean=mean, std=std, name="W")
                # if only mean was supplied
                elif mean:
                    W = get_weights_gaussian(shape=shape, mean=mean, name="W")
                # if only std was supplied
                elif std:
                    W = get_weights_gaussian(shape=shape, std=std, name="W")
                # otherwise, use the defaults from the function
                else:
                    W = get_weights_gaussian(shape=shape, name="W")
            # if we are initializing weights from a uniform distribution
            elif self.args.get('weights_init').lower() == 'uniform':
                interval = self.args.get('weights_interval')
                if interval:
                    W = get_weights_uniform(shape=shape, interval=interval, name="W")
                else:
                    log.error("No interval provided for get_weights_uniform!")
                    raise TypeError("No interval provided for get_weights_uniform!")
            # otherwise not implemented
            else:
                log.error("Did not recognize weights_init %s! Pleas try gaussian or uniform" % str(self.args.get('weights_init')))
                raise NotImplementedError("Did not recognize weights_init %s! Pleas try gaussian or uniform" % str(self.args.get('weights_init')))

            bias_init = self.args.get('bias_init')
            if bias_init:
                b = get_bias(shape=shape[1], name="b", init_values=bias_init)
            else:
                b = get_bias(shape=shape[1], name="b")

        # Finally have the two parameters!
        self.params = [W, b]

        # Grab the activation function to use
        activation_func = get_activation_function(self.args.get('activation'))

        # Here is the meat of the computation transforming input -> output
        self.output = activation_func(T.dot(self.input, W) + b)

        log.debug("Initialized a simple fully-connected layer with input shape: %s" % str(shape))

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