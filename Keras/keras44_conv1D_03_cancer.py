###############  Conv1D 유방암 버전  ################

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Conv1D, Flatten
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.saving.save import load_model

#1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)

x = x.reshape(569, 30, 1)    #스플릿전에 리쉐이프해야한다

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size =0.8, shuffle=True, random_state = 46)
             

#2. 모델구성
model = Sequential()
model.add(Conv1D(200,10, activation='relu', input_shape=(30, 1))) ##행은 넣지않는다
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(80, activation='linear'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))  



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, 
          #validation_data=(x_val, y_val))
          validation_split=0.3)

#validation_split을 사용하면 굳이 위에 데이터 정제작업에서 스플릿안해도 된다

#4. 평가, 예측
start = time.time()

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

end = time.time() - start

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)
print('걸린시간:', round(end,3), '초')