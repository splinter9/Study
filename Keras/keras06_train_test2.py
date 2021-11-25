import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

##과제## train과 test 비율을 8:2으로 분리하시오.
##리스트 슬라이싱 사용

'''케이스1) 순서대로, 훈련의 정확성 의심 - 과적합위험
x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:] '''

##케이스2) 랜덤하게, 훈련의 정확성 향상
x_train = x
x_test = x
y_train = y
y_test = y


 

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(50))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)


##결과##
'''
loss :  0.0010770537192001939
11의 예측값 :  [[11.037981]]

loss :  4.547473508864641e-13
11의 예측값 :  [[10.999999]]

loss :  3.637978807091713e-12
11의 예측값 :  [[11.]]

loss :  9.094947017729282e-13
11의 예측값 :  [[11.000001]]

loss :  4.297589839552529e-08
11의 예측값 :  [[10.999734]]
'''

