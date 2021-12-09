import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target 


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

print(x_train.shape, x_test.shape) # (464809, 54) (116203, 54)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(464809, 9, 6, 1)
x_test = scaler.fit_transform(x_test).reshape(116203, 9, 6, 1)

import pandas as pd
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(1, 1), padding='same', input_shape=(9, 6, 1)))
model.add(Flatten())
model.add(Dense(3))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)
