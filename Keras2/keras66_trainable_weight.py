from pyexpat import model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
'''

print(model.weights)
print('==============================================')
print(model.trainable_weights)
print('==============================================')
print(len(model.weights))
print(len(model.trainable_weights))
print('==============================================')

model.trainable=False
print(len(model.weights))
print(len(model.trainable_weights))
model.summary()
model.complie(loss='mse', optimizer='adam')
model.fit(x,y,batch_size=1,epochs=100)


# 전이 학습은 남이 잘만들어 놓은 모델을 갖다 쓰는 개념
