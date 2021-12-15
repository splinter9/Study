###############  LSMT 케글바이크 버전  ################
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
# sqrt : 제곱근

#1. 데이터 
path = './_data/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
print(submit_file.columns)    # ['datetime', 'count']


x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

# print(x.columns) 

'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
'''   
# print(x.shape)      # (10886, 8)
y = train['count']
# print(y.shape)      # (10886,)
# print(y)


print(x.shape, y.shape) #(10886, 8) (10886,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 36)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_train=x_train.reshape(x_train.shape[0], 8,1)
x_test=x_test.reshape(x_test.shape[0],8,1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성

model = Sequential()
model.add(LSTM(18, input_shape=(8,1))) 
model.add(Dense(17,activation='linear'))
model.add(Dense(15,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=30, batch_size=10,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)
