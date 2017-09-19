"""
This module gives an implementation of the Generative Stochastic Network model.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN
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
from keras.layers import Dense, Add
from keras.layers.noise import GaussianNoise
from .util_layers import SaltAndPepper


class GSN(object):
	def __init__(self, input, layers=[1500, 1500, 1500], walkbacks=5, hidden_act='relu', visible_act='sigmoid',
	             input_noise=.2, hidden_noise=2, input_sampling=True):
		self.input = input
		self.visible_act = visible_act
		self.hidden_act = hidden_act
		if visible_act is 'sigmoid':
			self.input_noise = SaltAndPepper(input_noise)
		else:
			self.input_noise = GaussianNoise(input_noise)

		self.hidden_noise = GaussianNoise(hidden_noise)

		recon_layer_sizes = layers[:-1][::-1] + [self.input._keras_shape[-1]]
		self.encoding_layers = []
		for size in layers:
			self.encoding_layers.append(Dense(size, activation=hidden_act))

		self.decoding_layers = []
		for i, size in enumerate(recon_layer_sizes):
			if i == len(layers) - 1:
				self.decoding_layers.append(Dense(size, activation=visible_act))
			else:
				self.decoding_layers.append(Dense(size, activation=hidden_act))

		self.walkbacks = walkbacks
		self.input_sampling = input_sampling

	def build_gsn(self):
		# corrupt input
		ins = self.input_noise(self.input)
		p_x_chain = []
		hiddens = [None] * len(self.encoding_layers)
		for walkback in range(self.walkbacks):
			for i in range(len(hiddens)):
				# first hidden layer and update pX chain
				if i == 0:



			# update even layers
			for i in range(0, len(hiddens), 2):
				pass
			# update odd layers and sample
			for i in range(1, len(hiddens), 2):
				pass
		return p_x_chain, hiddens


