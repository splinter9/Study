from pickletools import optimize
import numpy as np
from sklearn.model_selection import learning_curve

#1. DATA
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2.MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, SGD, Nadam 

learning_rate = 0.001 #러닝레이트에 따라 값이 크게 달라진다

# optimizer = Adam(learning_rate=0.0001)
# optimizer = Adam(learning_rate=learning_rate)
# optimizer = Adadelta(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
optimizer = Adamax(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
# optimizer = Nadam(learning_rate=learning_rate)
#각 옵티마이저별로 결과값 비교해볼것


# model.compile(loss='mae', optimizer='adam', metrics=['mse'])
model.compile(loss='mae', optimizer=optimizer)
model.fit(x,y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
y_predict = model.predict([11])

print('loss:', round(loss,4), 'lr:', learning_rate, '결과물:', y_predict)


'''
learnig_rate 0.1

0.1 / Adam
loss: 1.309 lr: 0.1 결과물: [[10.163114]]

0.1 / Adadelta
loss: 1.3841 lr: 0.1 결과물: [[12.31967]]

0.1 / Adagrad
loss: 1.5669 lr: 0.1 결과물: [[11.430144]]

0.1 / Adamax
loss: 4.4503 lr: 0.1 결과물: [[0.69752145]]

0.1 / RMSprop
loss: 109563.1406 lr: 0.1 결과물: [[-267234.88]]

0.1 / SGD
loss: nan lr: 0.1 결과물: [[nan]]

0.1 / Nadam
loss: 1.8453 lr: 0.1 결과물: [[7.2602406]]
'''

'''
learnig_rate 0.001

0.001 / Adam
loss: 1.2976 lr: 0.001 결과물: [[11.421438]]

0.001 / Adadelta
loss: 1.1101 lr: 0.001 결과물: [[10.89686]]

0.001 / Adagrad
loss: 1.109 lr: 0.001 결과물: [[10.976098]]

0.001 / Adamax
loss: 1.1228 lr: 0.001 결과물: [[10.949704]]

0.001 / RMSprop


0.001 / SGD


0.001 / Nadam

'''