
#1. R2를 0.8 이상 만들어라

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_boston
dataset = load_boston()

#1. 데이터
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape)  #(506, 13) dim=13
print(y.shape)  #(506,)

print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=41)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(8))
model.add(Dense(22))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  #로스값은 훈련에 영향을 주지 않는다
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)


'''
loss :  17.651206970214844
r2값은:  0.7863492753966558

loss :  19.576597213745117
r2값은:  0.7630442430959832

'''

















'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.7, shuffle=True, random_state=65)

print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x, y)  #로스값은 훈련에 영향을 주지 않는다, 결과니까...
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2값은: ', r2)
'''
