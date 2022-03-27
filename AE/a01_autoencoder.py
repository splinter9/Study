# 앞뒤가 똑같은

from base64 import encode
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float'/255)


#2. MODEL
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))

encoded = Dense(16, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(154, activation='relu')(input_img)  # PCA 0.95 지점
encoded = Dense(486, activation='relu')(input_img)  # PCA 0.999 지점
encoded = Dense(1024, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)


#3. COMPILE
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, )

#4. EVALUTATE
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    