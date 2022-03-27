# 앞뒤가 똑같은
# 딥하게 구성
# 2개의 모델을 구성하는데


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

def autoencoder1(hidden1):
    model = Sequential([
    Dense(units=hidden1, input_shape=(784, ),
                    activation='relu'),
    Dense(units=784, activation='sigmoid')
    ])
    return model

def autoencoder2(hidden1, hidden2):
    model = Sequential([
    Dense(units=hidden2, input_shape=(784, ),
                    activation='relu'),
    Dense(units=784, activation='sigmoid')
    ])
    return model

model = autoencoder1(hidden_layer_size=32)
model = autoencoder2(hidden_layer_size=32)



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

