import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = load_wine()
x = datasets.data # (178, 13)
y = datasets.target # (178,)

import pandas as pd
x_refine = pd.DataFrame(x, columns=datasets.feature_names)
x_cnn = x_refine.drop(columns=['flavanoids'], axis=1) # (178, 12)
x_cnn = x_cnn.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x_cnn, y, train_size=0.8, random_state=1)

print(x_train.shape, x_test.shape) # (142, 12) (36, 12)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(142, 3, 4, 1)
x_test = scaler.fit_transform(x_test).reshape(36, 3, 4, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(1, 1), padding='same', input_shape=(3, 4, 1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)