###############  conv1D 보스턴 버전  ################
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Bidirectional, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_boston



#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)

x = x.reshape(506, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)



#2. 모델구성
model = Sequential()
model.add(Conv1D(200,5, input_shape=(13,1))) ##행은 넣지않는다
model.add(Dense(150, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(80, activation='linear'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))  ##플랫튼 필요없이 덴스로 입력가능



#3. 컴파일, 훈련
start = time.time()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1)

end = time.time() - start


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

#from sklearn.metrics import r2_score
#r2 = r2_score(y_test, y_predict)        # r2 보다 loss가 더 신뢰성
#print('r2 스코어 : ', r2)

print('걸린시간:', round(end,3), '초')
