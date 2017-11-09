import numpy as np
from PIL import Image
from keras.backend import variable
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model

from src.callbacks.decay_functions import get_decay_function
from src.callbacks.schedule import Schedule
from src.models.dae import dae

if __name__ == '__main__':
    # make the model
    input = Input(shape=[784, ])
    init_noise = 0.5
    noise = variable(value=init_noise, dtype="float32")
    noise_decay = get_decay_function('montreal', parameter=noise, initial=init_noise, reduction_factor=.7)
    noise_schedule = Schedule(noise_decay, schedule_fn=lambda variable, _: variable.decay())
    model_layers = dae(input, layers=[1000], hidden_act='relu', visible_act='sigmoid', noise=noise)
    DAE = Model(inputs=input, outputs=model_layers[-1])
    DAE.compile(optimizer='adam', loss='binary_crossentropy')

    # grab the data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
    print("x shape:", x_train.shape)

    DAE.fit(x=x_train, y=x_train, batch_size=128, epochs=10, shuffle=True, callbacks=[noise_schedule])

    reconstructed = DAE.predict(x_test)

    n = 20
    for i in range(n):
        test = (x_test[i].reshape(28, 28) * 255).astype('uint8')
        pred = (reconstructed[i].reshape(28, 28) * 255).astype('uint8')
        test_img = Image.fromarray(test)
        test_img.save("x_test_" + str(i) + ".jpg")
        pred_img = Image.fromarray(pred)
        pred_img.save("x_recon_" + str(i) + ".jpg")
