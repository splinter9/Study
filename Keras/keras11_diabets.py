##########과제   R2 0.62 이상, R2 0.8 이상 ##############

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.8, shuffle=True, random_state=49)


#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dense(19))
model.add(Dense(18))
model.add(Dense(17))
model.add(Dense(16))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))

model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=0)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

'''
loss :  2035.3922119140625
r2값은:  0.6180043234073688

loss :  2102.294677734375
r2값은:  0.6054483108286682

loss :  2102.022705078125
r2값은:  0.6054992975118771

loss :  2075.049072265625
r2값은:  0.610561645129277

loss :  2053.822265625
r2값은:  0.6145453764384312
'''