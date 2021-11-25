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
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss값은 작을수록 좋다, loss에 값을 감축시키는 역할을 해줌(optimizer)
model.fit(x, y, epochs=100, batch_size=1) #epochs 훈련횟수 #batch 한번의 batch마다 주어지는 데이터 샘플 size, batch는 나눠진 데이터셋 #interation은 epoch를 나누어서 실행하는 횟수

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)


"""
loss :  8.8690927668722e-07
4의 예측값 :  [[4.002]]
레거시안러닝보다 딥러닝이 훨씬 효율적이고 값이 좋게 나온다
"""

