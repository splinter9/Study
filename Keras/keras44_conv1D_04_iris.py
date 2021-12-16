###############  Conv1D 아이리스 버전  ################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_iris()
x = datasets.data
y= datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(x.shape, y.shape)  #(150, 4) (150, 3)
print(np.unique(y))  #[0. 1.]


x=x.reshape(150,4,1)  #스플릿전에 리쉐이프해야한다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성

model = Sequential()
model.add(Conv1D(10,2, activation='linear', input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

import time
start = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


end = time.time() - start


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('acccuracy: ', loss[1])    


results = model.predict(x_test[:7])
print(y_test[:7])
print(results)
print("훈련시간:" , round(end, 3), '초')
