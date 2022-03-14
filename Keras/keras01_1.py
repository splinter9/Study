#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 이 데이터로 훈련해서 최소의 Loss값을 구해보자


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # 인풋값 1


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss값은 작을수록 좋다, loss에 값을 감축시키는 역할을 해줌(optimizer)
model.fit(x, y, epochs=1000, batch_size=1) #epochs 훈련량


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4]) # 네번째 값 
print('4의 예측값 : ', result)


"""
loss :  0.0
4의 예측값 :  [[4.]]
"""
