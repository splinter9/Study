import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,  2,   3,   4,   5,   6,   7,   8,   9,   10 ], 
              [1,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10, 9,   8,   7,   6,   5,   4,   3,   2,   1  ]])
y = np.array([ 11, 12,  13,  14,  15,  16,  17,  18,  19,  20 ])
x = np.transpose(x)

# x = x.T 
# x = np.transpose(x)    #행과 열이 바뀐다
# x = x.reshape(10,2)    #테이터 배열이 바뀐다
               

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3, 1]])  # x의 인풋 디멘션과 열이 같아야 한다 #열우선행무시
print('[10, 1.3, 1]의 예측값 :', y_predict)


'''
loss :  0.00010637744708219543
[10, 1.3, 1]의 예측값 : [[20.00247]]

loss :  0.00024024902086239308
[10, 1.3, 1]의 예측값 : [[20.002653]]

loss :  0.00023060510284267366
[10, 1.3, 1]의 예측값 : [[20.007315]]
'''
