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
print(x.shape, y.shape)  ## (13, 3) (13,)

## input_shape = (batch_size, timesteps, feature)
## input_shape = (batch_sizeg 행, timesteps 열, feature 자르는 갯수)
x = x.reshape(13, 3, 1) ## 4행, 3열, 1개씩 자름


#2. 모델구성
model = Sequential()
model.add(LSTM(220, activation='relu', input_shape=(3, 1))) ##행은 넣지않는다
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
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=1000)
end = time.time() - start

#4. 평가 예측
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]]) # y shape (4,)을 (1,3,1)로 바꿔줌



print(result)
print('걸린시간:', round(end,3), '초')

########### LSTM 결과#############
#[[80.01927]]
#[[80.8529]]
#[[80.47537]]
#[[80.30896]]
#[[80.22688]]
#[[79.945076]]
#[[80.949326]]
#[[79.81548]]  걸린시간: 4.329 초


########### GRU 결과 #############
#[[79.67575]]
#[[80.08067]]
#[[80.0981]]
#[[79.778595]]
#[[79.89745]]
#[[79.91288]]  걸린시간: 4.288 초
