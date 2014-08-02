import argparse
import Story1_old
import Story1

def main():
    parser = argparse.ArgumentParser()
    # Add options here

    parser.add_argument('--layers', type=int, default=3) # number of hidden layers
    parser.add_argument('--walkbacks', type=int, default=5) # number of walkbacks
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--save_frequency', type=int, default=10) #number of epochs between parameters being saved
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=2)
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4) #default=0.4
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--annealing', type=float, default=0.995)
    parser.add_argument('--hidden_size', type=int, default=1500)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--regularize_weight', type=float, default=0)
    parser.add_argument('--max_iterations', type=int, default=3)
   
    # argparse does not deal with bool 
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1)
    parser.add_argument('--input_sampling', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--continue_training', type=int, default=0) #default=0
    
    args = parser.parse_args()
       
    Story1_old.experiment(args,'../outputs/model_1/old/')
    
if __name__ == '__main__':
    main()
