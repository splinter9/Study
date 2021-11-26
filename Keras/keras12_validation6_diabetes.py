##########  validation train test4 diabets  ##############
#1. R2를 0.8 이상 만들어라


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_diabetes
datasets = load_diabetes()

#1. 데이터

x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)  #(506, 13) dim=13
print(y.shape)  #(506,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(80))
model.add(Dense(65))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2, 
          #validation_data=(x_val, y_val))
          validation_split=0.1)

#validation_split을 사용하면 굳이 위에 데이터 정제작업에서 스플릿안해도 된다

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

