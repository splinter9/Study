import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time


#1.데이터 
x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #덩어리로 자르는것을 timesteps라고 한다

x_predict = np.array([50,60,70])
x = x.reshape(13,3,1)

'''
#2. 모델구성
model = Sequential()
model.add(LSTM(300, return_sequences=True, input_shape=(3, 1))) #(N,3,1) -> (N,3,10)
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(150, return_sequences=True, activation='relu'))
model.add(LSTM(180, return_sequences=True, activation='relu'))
model.add(LSTM(80, return_sequences=False, activation='linear'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
start = time.time()
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=1000)
end = time.time() - start


#4. 평가 예측
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]]) # y shape (4,)을 (1,3,1)로 바꿔줌


print(result)
print('걸린시간:', round(end,3), '초')
'''