from base64 import encode
from typing import Sequence
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


#2. MODEL
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                    activation='relu')),
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08= autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)


model_01.compile(loss = 'binary_crossentropy', optimizer='adam')
model_01.fit(x_train, x_train , epochs=10)

model_02.compile(loss = 'binary_crossentropy', optimizer='adam')
model_02.fit(x_train, x_train , epochs=10)

model_04.compile(loss = 'binary_crossentropy', optimizer='adam')
model_04.fit(x_train, x_train , epochs=10)

model_08.compile(loss = 'binary_crossentropy', optimizer='adam')
model_08.fit(x_train, x_train , epochs=10)

model_16.compile(loss = 'binary_crossentropy', optimizer='adam')
model_16.fit(x_train, x_train , epochs=10)

model_32.compile(loss = 'binary_crossentropy', optimizer='adam')
model_32.fit(x_train, x_train , epochs=10)


output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

from matplotlib import pyplot as plt
import random 

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output_01.shape([0]), 5))
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].np.reshape(28,28),
                  cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.show()







'''
#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, x_train , epochs=20)


#4. PREDICT
output = model.predict(x_test)

import matplotlib.pyplot as plt
from matplotlib import pyplot
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
      plt.pyplot.subplot(2, 5, figsize=(20, 7))


random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[1]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()
'''

