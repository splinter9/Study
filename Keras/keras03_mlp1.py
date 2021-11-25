import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2,   3,   4,   5,   6,   7,   8,   9,   10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.transpose(x)

# x = x.T 
# x = np.transpose(x)    #행과 열이 바뀐다
# x = x.reshape(10,2)    #테이터 배열이 바뀐다
               

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=2))
model.add(Dense(50))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3]])  # x의 인풋 디멘션과 열이 같아야 한다 #열우선행무시
print('[10, 1.3]의 예측값 :', y_predict)

'''
loss :  0.0004832086560782045
[10, 1.3]의 예측값 : [[20.002756]]
'''
