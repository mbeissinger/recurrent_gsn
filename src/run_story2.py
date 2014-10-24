import argparse
import Story2e_untied_walkbacks as Story2

def main():
    parser = argparse.ArgumentParser()
    # Add options here

    parser.add_argument('--layers', type=int, default=3) # number of hidden layers
    parser.add_argument('--walkbacks', type=int, default=5) # number of walkbacks
#     parser.add_argument('--recurrent_walkbacks', type=int, default=0) # number of recurrent walkbacks
    parser.add_argument('--n_epoch', type=int, default=600)
    parser.add_argument('--save_frequency', type=int, default=10) #number of epochs between parameters being saved
    parser.add_argument('--early_stop_threshold', type=float, default=0.9995)
    parser.add_argument('--early_stop_length', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=106) #batch size here to speed up training - doesn't work in practice unless you know the sequence length. In my case it is 10, so using 10*10+walkbacks+1 allows the continuation of the recurrent hidden variables.
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2) #default=2
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.05) #default=0.4
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--hidden_size', type=int, default=1500)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--dataset', type=str, default='MNIST_1')
    parser.add_argument('--data_path', type=str, default='../data/')
#     parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--regularize_weight', type=float, default=0)
    parser.add_argument('--max_iterations', type=int, default=1)
   
    # argparse does not deal with bool 
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1) #default=1
    parser.add_argument('--input_sampling', type=int, default=0) #default=1  In this case we don't need input sampling because it is not a gibbs generative
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--continue_training', type=int, default=0) #default=0
    
    args = parser.parse_args()
       
    # RUN STORY 2
    #Story2.experiment(args, '../outputs/model_2/')
    args.dataset = "MNIST_2"
    Story2.experiment(args, '../outputs/model_2/')
    args.dataset = "MNIST_3"
    Story2.experiment(args, '../outputs/model_2/')
    
if __name__ == '__main__':
    main()
