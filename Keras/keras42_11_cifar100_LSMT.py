###############  LSMT CIFAR100 버전  ################
from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout,Conv2D, Flatten,MaxPooling2D 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) =cifar100.load_data()
print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32,96) ##32*3
x_test = x_test.reshape(10000,32,96)
print(x_train.shape)

# print(np.unique(y_train, return_counts=True))   #  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# x= x_train

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)   
#print(y)
#print(y_train.shape)  
y_test = to_categorical(y_test)
#print(y_test.shape)

# scaler= StandardScaler()
# x_train= x_train.reshape(50000,-1)      # 4차원 (50000,32,32,3)을 가로로 1자로 쫙펴준다.  행 세로 열 가로   (50000,3072)
# x_test = x_test.reshape(10000,-1)

# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train= x_train.reshape(50000,32,32,3)
# x_test= x_test.reshape(10000,32,32,3)


model = Sequential()
model.add(LSTM(20, input_shape=(32, 96)))
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)