###############  LSMT 와인 버전  ################
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape)  #(178, 13) (178,)
print(np.unique(y))  #[0 1 2]

y=to_categorical(y)
print(y)
print(y.shape)  #(178, 3)

x=x.reshape(178,13,1)  #스플릿전에 리쉐이프해야한다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential()
model.add(LSTM(50, activation='linear', input_shape=(13,1)))
model.add(Dense(55))
model.add(Dense(60, activation='relu'))
model.add(Dense(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  



loss=model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:5])
print(y_test[:5])
print(results)
