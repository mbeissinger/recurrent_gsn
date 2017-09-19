import argparse

import theano

from rnngsn import RNN_GSN
from utils import data_tools as data
from utils import logger as log
from utils.utils import raise_to_list


def main():
    parser = argparse.ArgumentParser()
    # Add options here

    # GSN settings
    parser.add_argument('--layers', type=int, default=3) # number of hidden layers
    parser.add_argument('--walkbacks', type=int, default=5) # number of walkbacks
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--hidden_act', type=str, default='tanh')
    parser.add_argument('--visible_act', type=str, default='sigmoid')
    
    # recurrent settings
    parser.add_argument('--recurrent_hidden_size', type=int, default=100)
    parser.add_argument('--recurrent_hidden_act', type=str, default='tanh')
    
    # training
    parser.add_argument('--initialize_gsn', type=int, default=0) # whether or not to train a strict GSN first to initialize the weights and biases
    parser.add_argument('--cost_funct', type=str, default='binary_crossentropy') # the cost function for training
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--gsn_batch_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100) # max length of sequence to consider
    parser.add_argument('--save_frequency', type=int, default=10) #number of epochs between parameters being saved
    parser.add_argument('--early_stop_threshold', type=float, default=0.9996) #0.9995
    parser.add_argument('--early_stop_length', type=int, default=30)
    parser.add_argument('--hessian_free', type=int, default=0) # boolean for whether or not to use Hessian-free training for RNN-GSN
    
    # noise
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2)
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4) #default=0.4
    
    # hyper parameters
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--noise_annealing', type=float, default=1) #default=1 for no noise schedule
    parser.add_argument('--regularize_weight', type=float, default=0)
    
    # data
    parser.add_argument('--dataset', type=str, default='MNIST_4')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--outdir_base', type=str, default='../outputs/rnn_gsn/')
   
    # argparse does not deal with booleans
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1)
    parser.add_argument('--input_sampling', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--continue_training', type=int, default=0) #default=0
    
    return parser.parse_args()
    
def create_rnngsn(args):
    (train,_), (valid,_), (test,_) = data.load_datasets(args.dataset, args.data_path)
        
    train_X = raise_to_list(train)
    valid_X = raise_to_list(valid)
    test_X = raise_to_list(test)
    train_X = [theano.shared(t, borrow=True) for t in train_X]
    valid_X = [theano.shared(v, borrow=True) for v in valid_X]
    test_X  = [theano.shared(t, borrow=True) for t in test_X]
        
        
    args.is_image = True
    
    args.output_path = args.outdir_base + args.dataset
    
    logger = log.Logger(args.output_path)
    
    rnngsn = RNN_GSN(train_X=train_X, valid_X=valid_X, test_X=test_X, args=vars(args), logger=logger)
    rnngsn.train()
    # rnngsn.load_params('nottingham_params.pkl')
    # rnngsn.gen_10k_samples()
    
    # sample_paths = ['samples_test'+str(i)+'.npy' for i in range(len(test_X))]
        
    ## parzen
    # print 'Evaluating parzen window'
    # import utils.likelihood_estimation as ll
    # ll.main(0.20,'nottingham','../data/',sample_paths) 
    
    
if __name__ == '__main__':
    args = main()
    create_rnngsn(args)
