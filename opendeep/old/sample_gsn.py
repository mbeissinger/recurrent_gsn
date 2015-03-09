import argparse
import numpy
import os

from opendeep.models.multi_layer.generative_stochastic_network import GSN
from utils import data_tools as data
import utils.logger as log
from utils.utils import load_from_config




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
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_frequency', type=int, default=5) #number of epochs between parameters being saved
    parser.add_argument('--early_stop_threshold', type=float, default=0.9995)
    parser.add_argument('--early_stop_length', type=int, default=30) #the patience number of epochs
    
    # noise
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2) #default=2
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4) #default=0.4
    
    # hyper parameters
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--noise_annealing', type=float, default=1)
    
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
#     gsn.train()
    
    gsn.load_params('gsn_params_mnist.pkl')
    gsn.gen_10k_samples()
    # parzen
    print 'Evaluating parzen window'
    import utils.likelihood_estimation as ll
    ll.main(0.20,'mnist','../data/','samples.npy') 
    
#     
#     _, h_samples = gsn.sample(train_X[0:1].eval(), 100000, 100)
#     
#     print ll.CSL(h_samples, test_X.get_value(), gsn)
        
#     logger.log(str(ll.CSL(h_samples, test_X.get_value(), gsn)))

if __name__ == '__main__':
    main()