##########  모델별로 저장하기  ##############



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import History
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                 train_size=0.8, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(22, input_dim=13))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

model.save("./_save/keras25_1_save_model.h5")


#3. 컴파일, 훈련
import time
start = time.time()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          #validation_data=(x_val, y_val))
          validation_split=0.3)

end = time.time() - start
print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)








