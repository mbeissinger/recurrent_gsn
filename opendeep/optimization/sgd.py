'''
Generic stochastic gradient descent optimization
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

import logging

import numpy
import theano
import theano.tensor as T

log = logging.getLogger(__name__)

from optimizer import Optimizer

class SGD(Optimizer):
    '''
    Stochastic gradient descent for training a model - includes early stopping
    '''
    #TODO: add conjugate gradient?

    def __init__(self, model, train_data, train_targets, valid_data=None, valid_targets=None, test_data=None, test_targets=None, config=None):
        pass

    def train(self, train_X=None, valid_X=None, test_X=None, continue_training=False):
        log.maybeLog(self.logger, "\nTraining---------\n")
        if train_X is None:
            log.maybeLog(self.logger, "Training using data given during initialization of GSN class.\n")
            if self.train_X is None:
                log.maybeLog(self.logger, "\nPlease provide a training dataset!\n")
                raise AssertionError("Please provide a training dataset!")
        else:
            log.maybeLog(self.logger, "Training using data provided to training function.\n")
            self.train_X = train_X
        if valid_X is not None:
            self.valid_X = valid_X
        if test_X is not None:
            self.test_X  = test_X

        # Input data
        self.train_X = raise_to_list(self.train_X)
        self.valid_X = raise_to_list(self.valid_X)
        self.test_X  = raise_to_list(self.test_X)

        ########################################################
        # Compile training functions to use indexing for speed #
        ########################################################
#         log.maybeLog(self.logger, "Compiling training functions...")
#         t = time.time()
#         self.compile_train_functions(train_X, valid_X, test_X)
#         log.maybeLog(self.logger, "Compiling done. Took "+make_time_units_string(time.time() - t)+".\n")



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

        x_shape = self.train_X[0].get_value(borrow=True).shape
        log.maybeLog(self.logger, ['train X size:',str(x_shape)])
        if self.valid_X is not None:
            vx_shape = self.valid_X[0].get_value(borrow=True).shape
            log.maybeLog(self.logger, ['valid X size:',str(vx_shape)])
        if self.test_X is not None:
            tx_shape = self.test_X[0].get_value(borrow=True).shape
            log.maybeLog(self.logger, ['test X size:',str(tx_shape)])

        if self.vis_init:
            self.bias_list[0].set_value(logit(numpy.clip(0.9,0.001,self.train_X[0].get_value(borrow=True).mean(axis=0))))

        start_time = time.time()

        while not STOP:
            counter += 1
            t = time.time()
            log.maybeAppend(self.logger, [counter,'\t'])

            #shuffle the data
#             data.shuffle_data(self.train_X)
#             data.shuffle_data(self.valid_X)
#             data.shuffle_data(self.test_X)

            #train
            train_costs = []
            for train_data in self.train_X:
                train_costs.extend(data.apply_cost_function_to_dataset(self.f_learn, train_data, self.batch_size))
#             train_costs = data.apply_indexed_cost_function_to_dataset(self.f_learn, x_shape[0], self.batch_size)
            log.maybeAppend(self.logger, ['Train:',trunc(numpy.mean(train_costs)), '\t'])

            #valid
            if self.valid_X is not None:
                valid_costs = []
                for valid_data in self.valid_X:
                    valid_costs.extend(data.apply_cost_function_to_dataset(self.f_cost, valid_data, self.batch_size))
#                 valid_costs = data.apply_indexed_cost_function_to_dataset(self.f_valid, vx_shape[0], self.batch_size)
                log.maybeAppend(self.logger, ['Valid:',trunc(numpy.mean(valid_costs)), '\t'])

            #test
            if self.test_X is not None:
                test_costs = []
                for test_data in self.test_X:
                    test_costs.extend(data.apply_cost_function_to_dataset(self.f_cost, test_data, self.batch_size))
#                 test_costs = data.apply_indexed_cost_function_to_dataset(self.f_test, tx_shape[0], self.batch_size)
                log.maybeAppend(self.logger, ['Test:',trunc(numpy.mean(test_costs)), '\t'])

            #check for early stopping
            if self.valid_X is not None:
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
                self.save_params(counter, self.params)

            timing = time.time() - t
            times.append(timing)

            log.maybeAppend(self.logger, 'time: '+make_time_units_string(timing)+'\t')

            log.maybeLog(self.logger, 'remaining: '+make_time_units_string((self.n_epoch - counter) * numpy.mean(times)))

            if (counter % self.save_frequency) == 0 or STOP is True:
                if self.is_image:
                    n_examples = 100
                    tests = self.test_X[0].get_value(borrow=True)[0:n_examples]
                    noisy_tests = self.f_noise(self.test_X[0].get_value(borrow=True)[0:n_examples])
                    _, reconstructed = self.f_recon(noisy_tests)
                    # Concatenate stuff if it is an image
                    stacked = numpy.vstack([numpy.vstack([tests[i*10 : (i+1)*10], noisy_tests[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
                    number_reconstruction = PIL.Image.fromarray(tile_raster_images(stacked, (self.image_height,self.image_width), (10,30)))

                    number_reconstruction.save(self.outdir+'gsn_image_reconstruction_epoch_'+str(counter)+'.png')

                    self.plot_samples(counter, "gsn", 1000)

                #save gsn_params
                self.save_params(counter, self.params)

            # ANNEAL!
            new_lr = self.learning_rate.get_value() * self.annealing
            self.learning_rate.set_value(new_lr)

#             new_hidden_sigma = self.hidden_add_noise_sigma.get_value() * self.noise_annealing
#             self.hidden_add_noise_sigma.set_value(new_hidden_sigma)

            new_salt_pepper = self.input_salt_and_pepper.get_value() * self.noise_annealing
            self.input_salt_and_pepper.set_value(new_salt_pepper)

        log.maybeLog(self.logger, "\n------------TOTAL GSN TRAIN TIME TOOK {0!s}---------\n\n".format(make_time_units_string(time.time()-start_time)))


        def sgd_optimizer(p, inputs, costs, train_set, lr=1e-4):
            '''SGD optimizer with a similar interface to hf_optimizer.'''

            g = [T.grad(costs[0], i) for i in p]
            updates = dict((i, i - lr*j) for i, j in zip(p, g))
            f = theano.function(inputs, costs, updates=updates)

            try:
                for u in xrange(1000):
                    cost = []
                    for i in train_set.iterate(True):
                        cost.append(f(*i))
                    print 'update %i, cost=' %u, numpy.mean(cost, axis=0)
                    sys.stdout.flush()

            except KeyboardInterrupt:
                print 'Training interrupted.'