import logging

from sklearn.feature_extraction.text import CountVectorizer

from opendeep.log.logger import config_root_logger
from datasets.util import FileParser
from opendeep.data.dataset import MemoryDataset
from opendeep.models.multi_layer.generative_stochastic_network import GSN

log = logging.getLogger(__name__)

def main():
    ########################################
    # Initialization things with arguments #
    ########################################
    config_root_logger()
    log.info("Creating a new GSN")

    dataparser = FileParser()
    data_x, data_y = dataparser.parse('widgets/widgets-filtered.csv')

    # vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='ascii', max_df=0.5, stop_words='english')
    vectorizer = CountVectorizer(strip_accents='ascii', binary=True, stop_words='english')
    data_x = vectorizer.fit_transform(data_x)
    data_x = data_x.todense()



    data = MemoryDataset(train_X=data_x, train_Y=data_y)

    config={# gsn parameters
            "layers": 2, # number of hidden layers to use
            "walkbacks": 4, # number of walkbacks (generally 2*layers) - need enough to have info from top layer propagate to visible layer
            "hidden_size": 1000,
            "visible_activation": 'sigmoid',
            "hidden_activation": 'tanh',
            "input_sampling": True,
            # train param
            "cost_function": 'binary_crossentropy',
            # noise parameters
            "noise_annealing": 1.0, #no noise schedule by default
            "add_noise": True,
            "noiseless_h1": True,
            "hidden_add_noise_sigma": 2,
            "input_salt_and_pepper": 0.4,
            # data parameters
            "output_path": 'outputs/gsn/bow/',
            "is_image": False,
            "vis_init": False}

    train_config = {"n_epoch": 1000,
               "batch_size": 100,
               "save_frequency": 10,
               "early_stop_threshold": .9995,
               "early_stop_length": 30,
               "learning_rate": 0.25,
               "lr_decay": "exponential",
               "lr_factor": .995,
               "annealing": 0.995,
               "momentum": 0.5,
               "unsupervised": True}

    gsn = GSN(config=config, dataset=data)

    # # Load initial weights and biases from file if testing
    # params_to_load = 'gsn_params.pkl'
    # if test_model:
    #     gsn.load_params(params_to_load)

    gsn.train(optimizer_config=train_config)



if __name__ == '__main__':
    main()