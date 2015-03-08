'''
General interface for creating a model - this is the same for simple layers, modules, and a full-blown model.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
import time
import os
import cPickle
# third party libraries
from theano.compat.python2x import OrderedDict
# internal references
from opendeep import function  # use this wrapper for theano.function - it removes errors when inputs aren't used.
from opendeep.utils.config import create_dictionary_like
from opendeep.utils.nnet import make_time_units_string
# from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.optimization.adadelta import AdaDelta # Use AdaDelta by default - safer than picking momentum for SGD
from opendeep.data.iterators.sequential import SequentialIterator

log = logging.getLogger(__name__)

class Model(object):
    '''
    Default interface for creating a model (or any sub-component of one).
    Think of it like legos.
    '''

    def __init__(self, config=None, defaults=None, input_hook=None, dataset=None):
        '''
        This sets up the model's configuration options into a self.args dictionary-like object.
        Further, it should establish all variables and parameters needed, and put them in the self object.

        Finally, the initialization should also include creating the computational graph, where you write the
        calculations transforming the input into the output. The end of this should include compiling the theano
        function (wrapped by opendeep.function) self.f_predict, which calculates the output given the input.

        :param config: a dictionary-like object containing all the necessary parameters for the model.
        Could be a dictionary-like object, .json file, or .yaml file.
        :param defaults: a dictionary-like object containing all the necessary default parameters for the model.
        Could be a dictionary-like object, .json file, or .yaml file.
        :return: None
        '''
        # make sure the config is like a dictionary
        config_dict = create_dictionary_like(config)
        # make sure the defaults is like a dictionary
        defaults_dict = create_dictionary_like(defaults)
        # override any default values with the config (after making sure they parsed correctly)
        if config_dict and defaults_dict:
            defaults_dict.update(config_dict)
        # set this update combination to the model arguments
        self.args = defaults_dict

        # print the arguments
        log.debug("ARGS: %s", str(self.args))

        # this establishes if the inputs should be set from a previous layer
        # input_hook is the input that should be used instead of initializing a new theano variable.
        self.input = input_hook

        # this is the dataset you will want to train on (given here so input size can be set automatically)
        self.dataset = dataset

    def get_inputs(self):
        '''
        This should return the input to the model's computation graph - what was used as the inputs= in the
        theano f_predict function created during __init__.

        :return: theano variable
        It should be the same as was passed to updates= in the self.f_predict function.
        '''
        # by default, try to return self.input
        if self.input is not None:
            return [self.input]
        else:
            log.critical("The Model %s does not have a get_inputs function (of self.input parameter)!", str(type(self)))
            raise NotImplementedError()

    def get_outputs(self):
        '''
        This returns the output of the model's computation - what is in the outputs= of f_predict.
        Used for hooking up layers together - setting the input to the previous layer's get_outputs.

        :return: theano variable
        It should be the same as was passed to outputs= in the self.f_predict function.
        '''
        # by default, try to return self.output
        if hasattr(self, 'output') and self.output is not None:
            return self.output
        else:
            log.critical("The Model %s does not have a get_outputs function (of self.output parameter)!", str(type(self)))
            raise NotImplementedError()

    def get_hiddens(self):
        '''
        This returns the hook to the hidden representation variables that are sometimes used as input hooks to other layers.

        :return: theano variable
        A single theano variable representing the inner representation of the model.
        '''
        log.critical("The Model %s does not have a get_hiddens function!", str(type(self)))
        raise NotImplementedError()

    def get_updates(self):
        '''
        This should return any theano updates from the model (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation.
        This is used with the optimizer to create the training function - the 'updates' part of the theano function.

        :return: updates from the theano computation for the model performed in __init__.
        It should be the same as was passed to updates= in the self.f_predict function.
        '''
        # by default, assume the model doesn't have updates - it's the job of the creator to return them in this method.
        return None

    def get_train_cost(self):
        '''
        This returns the expression that represents the cost given an input, which is used for the optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the optimizer object.

        :return: expression (theano tensor)
        an expression giving the cost of the model for training.
        '''
        log.critical("The Model %s does not have a get_train_cost function!", str(type(self)))
        raise NotImplementedError()

    def get_train_params(self):
        '''
        This returns a list of the model parameters to be trained by the optimizer.

        :return: list
        Model parameters to be trained.
        '''
        log.critical("The Model %s does not have get_train_params!", str(type(self)))
        raise NotImplementedError()

    def get_decay_params(self):
        '''
        This returns a list of the opendeep.utils.decay_functions on internal parameters to decay each time step.
        Used with the optimizer, for example, to decay things like GSN noise over time (noise scheduling).
        '''
        # by default, there are most likely no model params to decay during training.
        return []

    def get_monitors(self):
        '''
        :return: OrderedDict
        An OrderedDict of the names : theano variables to use as the monitors during training - the model variables you want to check
        periodically during training.
        '''

        # by default, return the cost expression used during training.
        return OrderedDict([('train_cost', self.get_train_cost())])

    def get_monitor_function(self):
        '''
        Creates a theano function returning the value of the monitor variables. This function will be called during
        the optimizer's train method to output monitor values on the train, valid, and test sets.

        :return: theano function
        '''
        if not hasattr(self, 'f_monitors'):
            log.info('Compiling f_monitor function for model %s...', str(type(self)))
            t = time.time()
            self.f_monitors = function(inputs  = self.get_inputs(),
                                       updates = self.get_updates(),
                                       outputs = self.get_monitors().values(),
                                       name    = 'f_monitors')
            log.info('f_monitor compilation took %s', make_time_units_string(time.time() - t))
        return self.f_monitors

    def generate_without_input(self, values=None):
        '''
        This should generate model outputs (if a generative model) without a specific input given. In most cases (like
        gsn), this means providing the model's hidden values. In other cases (like rbm), this means setting input to
        zero.

        :param values: list
        The values given to the model to start generating from.
        :return: theano variable
        The expression of the outputs from the starting point given with values.
        '''
        log.critical("The Model %s does not have a generate_without_input function!", str(type(self)))
        raise NotImplementedError()

    def get_params(self):
        '''
        This gives a list of the model parameters

        :return: list
        List of the model parameters (still as shared variables).
        '''
        # try to return self.params by default
        try:
            return self.params
        except AttributeError:
            log.critical("The Model %s does not have get_params (or a self.params variable)!!", str(type(self)))
            raise NotImplementedError()

    def get_param_values(self, borrow=False):
        '''
        This gives a list of the model parameters as actual values

        :return: list
        List of the model parameter values.
        '''
        # try to return self.params by default
        try:
            # try to return the .get_value() of the parameter if it is theano shared - otherwise just return it.
            return [param.get_value(borrow=borrow) if hasattr(param, 'get_value') else param for param in self.get_params()]
        except AttributeError:
            log.critical("The Model %s does not have get_params (or a self.params variable)!!", str(type(self)))
            raise NotImplementedError()

    def set_params(self, values, borrow=False):
        '''
        This sets the list of the model parameters as actual values

        :return: Boolean
        If the operation was successful
        '''
        try:
            # try to set the parameters found from self.get_params()
            params = self.get_params()
            assert len(params) == len(values), "Model %s params length %s while values length %s in set_params" % (str(type(self)), str(len(params)), str(len(values)))
            log.debug("setting model %s parameters to given values...", str(type(self)))
            for param, value in zip(params, values):
                param.set_value(value, borrow=borrow)
            log.debug("done setting parameters.", str(type(self)))
            return True
        except AttributeError:
            log.critical("The Model %s does not have get_params (or a self.params variable) (or some param doesn't have set_value()!!", str(type(self)))
            raise NotImplementedError()

    def load_params(self, param_file):
        '''
        This loads and sets the model paramaters found in the param_file (pickle file)

        :param param_file: filename of pickled params file
        :return: Boolean
        whether successful
        '''
        if os.path.isfile(param_file):
            _, extension = os.path.splitext(param_file)
            if extension.lower() is '.pickle' or extension.lower() is '.pkl':
                log.debug("loading model %s parameters from %s...", str(type(self)), str(param_file))
                with open(param_file, 'r') as f:
                    loaded_params = cPickle.load(f)
                self.set_params(loaded_params, borrow=False)
                return True
            else:
                log.error('Param file %s doesn\'t have a supported extension! Must be a pickle file with .pickle or .pkl', str(param_file))
                return False
        else:
            log.error('Param file %s couldn\'t be found!', str(param_file))
            return False

    def save_params(self, param_file):
        '''
        This saves the model paramaters to the param_file (pickle file)

        :param param_file: filename of pickled params file
        :return: Boolean
        whether successful
        '''
        params = self.get_param_values()
        log.debug('Saving model %s parameters to %s...', str(type(self)), str(param_file))
        with open(param_file, 'wb') as f:
            try:
                cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            except Exception:
                log.error("Some issue saving model %s parameters to %s!", str(type(self)), str(param_file))
                return False
            finally:
                f.close()
        return True

    def score(self, X):
        """
        (description from pylearn2)
        Compute a "score function" for this model, if this model has
        probabilistic semantics.
        Parameters
        ----------
        X : tensor_like
            A batch of i.i.d. examples with examples indexed along the
            first axis and features along the second. This is data on which
            the monitoring quantities will be calculated (e.g., a validation
            set).
        Returns
        -------
        score : tensor_like
            The gradient of the negative log probability of the model
            on the given data.
        Notes
        -----
        If the model implements a probability distribution on R^n,
        this method should return the gradient of the log probability
        of the batch with respect to V, or raise an exception explaining
        why this is not possible.
        """
        log.critical("The Model %s does not have a probabilistic score method! (i.e. p(X=x|H)). This is used for log-likelihoods.",
                     str(type(self)))
        raise NotImplementedError()

    def train(self, dataset, iterator_class=SequentialIterator, optimizer=None, optimizer_config=None, rng=None):
        '''
        The method to train this model, given a dataset object (and an optimizer).
        Use generic AdaDelta by default --
        Some information from Andrej Karpath:
        'In my own experience, Adagrad/Adadelta are "safer" because they don't depend so strongly on setting of learning rates
        (with Adadelta being slightly better), but well-tuned SGD+Momentum almost always converges faster and at better final
        values.' http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html

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
        # use AdaDelta by default
        if optimizer is None:
            optimizer = AdaDelta

        # use self's dataset if it exists - this was given during initialization
        if self.dataset:
            dataset = self.dataset

        assert callable(optimizer), "Model %s optimizer during train() is not a callable class" % str(type(self))
        optimizer = optimizer(model=self, dataset=dataset, iterator_class=iterator_class, config=optimizer_config, rng=rng)

        optimizer.train()

        self.save_params('trained_params.pkl')

    def get_lr_scalers(self):
        '''
        If you want to scale individual parameters' learning rates, create the dictionary here mapping parameter: scaling value
        :return: Dictionary
        '''
        return {}