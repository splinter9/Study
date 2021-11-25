from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.7, shuffle=True, random_state=65)

print(x_test)
print(y_test)'''

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=1) #verbose0,1,2,3~ 
end = time.time() - start
print("걸린시간:" , end)


'''
verbose
0 안보임
1 다보임
2 로스까지
3~ 에포크까지
'''
'''
#4. 평가, 예측
loss = model.evaluate(x, y)  #로스값은 훈련에 영향을 주지 않는다, 결과니까...
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2값은: ', r2)
'''
