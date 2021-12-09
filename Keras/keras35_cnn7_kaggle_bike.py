import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터 
path = "../_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
y = np.log1p(y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# print(x_train.shape, x_test.shape) # (8708, 8) (2178, 8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(8708, 4, 2, 1)
x_test = scaler.fit_transform(x_test).reshape(2178, 4, 2, 1)



#2. 모델구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(1, 1), padding='same', input_shape=(4, 2, 1)))
model.add(Flatten())
model.add(Dense(3))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=30, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=2, validation_split=0.2, callbacks=[es])

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test, y_pred))   
rmse = RMSE(y_test, y_pred) 
print("RMSE : ", rmse)