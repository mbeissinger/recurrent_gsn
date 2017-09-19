"""
Denoising autoencoder
"""
from keras.layers import Dense
from keras.layers.noise import GaussianNoise
from .util_layers import SaltAndPepper


def dae(input, layers=None, hidden_act='relu', visible_act='sigmoid', noise=.2):
	if layers is not None:
		recon_layers = layers[:-1][::-1] + [input._keras_shape[-1]]
		# corrupt input
		if visible_act is 'sigmoid':
			corrupted_input = SaltAndPepper(noise)(input)
		else:
			corrupted_input = GaussianNoise(noise)(input)
		# encode corrupted input
		model_layers = [corrupted_input]
		_h = corrupted_input
		for size in layers:
			_h = Dense(size, activation=hidden_act)(_h)
			model_layers.append(_h)
		# decode encoding
		_reconstructed = _h
		for i, size in enumerate(recon_layers):
			if i == len(layers)-1:
				_reconstructed = Dense(size, activation=visible_act)(_reconstructed)
			else:
				_reconstructed = Dense(size, activation=hidden_act)(_reconstructed)
			model_layers.append(_reconstructed)
		return model_layers
	return None
