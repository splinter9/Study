#R2(결정계수)가 무엇인지 찿아라 (통계학에서 말하는 상관계수의 제곱과는 의미가 다르다)

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9, 3, 8,12,13,17, 8,14,21,23,41,25])

from sklearn.metrics import r2_score
R2 = r2_score(x, y)
print('r2값은: ', R2)
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9, 8, 8,12,13,17, 8,14,21,23,23,25])

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(30))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  #로스값은 훈련에 영향을 주지 않는다
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

#plt.scatter(x,y)  #점찍기
#plt.plot(x, y_predict) #선긋기
#plt.show()

