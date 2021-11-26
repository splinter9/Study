##########  validation  ##############

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array(range(17))
y = np.array(range(17))
x_train = x[:10]
x_test = x[10:17]
y_train = y[:10]
y_test = y[10:17]
x_val = x[:-3]
y_val = y[:-3] 


#x_train = np.array(range(1,11))
#y_train = np.array(range(1,11))
#x_test = np.array([11,12,13])
#y_test = np.array([11,12,13])
#x_val = np.array([14,15,16])
#y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print('17의 예측값: ', y_predict)

'''
Epoch 100/100
10/10 [==============================] - 0s 2ms/step - loss: 1.0936e-04 - val_loss: 7.4747e-04
1/1 [==============================] - 0s 62ms/step - loss: 3.1266e-04
loss :  0.0003126629744656384
17의 예측값:  [[16.966303]]

#로스값과 발로스값의 차이는 모델의 신뢰성을 보여줌
#val_loss를 더 신뢰할 수 있다
#loss는 과접합되어있다고 보임
'''

