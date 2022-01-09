#########################################################
############### 지역별 SOHO 폐업률 예측 LSTM#################
#########################################################  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy

#1. 데이터


path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA_T.csv")


x = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO','MED_ARR_AMT','CLSD_CNT'],axis=1)
y = dataset['CLSD_CNT']

x, y = x.values, y.values
y = np.log1p(y)
x = x.reshape(187, 32, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)


def check_missing_col(dataframe):
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'총 {missing_values}개의 결측치가 존재합니다.')

        if i == len(dataframe.columns) - 1 and counted_missing_col == 0:
            print('결측치가 존재하지 않습니다') #결측치가 존재하지 않습니다

check_missing_col(dataset)

# scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# #scaler = MaxAbsScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

print(x.shape, y.shape) #(187, 33) (187,)



#2. 모델구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(32,1))) 
model.add(Flatten())
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
#model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='linear'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련

import time
start = time.time()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

end = time.time() - start

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss:", loss)


y_predict = model.predict(x_test)
print('예측값 : ', y_predict[0:18])
 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)        # r2 보다 loss가 더 신뢰성높음??
print('r2 스코어 : ', r2)

print('걸린시간:', round(end,3), '초')


'''
loss :  642217.5625
예측값 :  [623.80035]
걸린시간: 896.292 초

loss :  0.00017904820560943335
예측값 :  [50820.453]
걸린시간: 958.239 초

loss: 234045.4375
예측값 :  [87293.06]
r2 스코어 :  -125693.659005371
걸린시간: 45.989 초




'''




'''
#le = LabelEncoder()
#y = to_categorical(y)
print(x.shape)      #(187, 40)
print(y.shape)      #(187, 37286)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)  

# #데이터표준화
# mean = np.mean(x_train, axis=0)
# std = np.sdt(x_train, axis=0)

# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std

print(x_train.shape, y_train.shape) #(130, 40) (130, 37286)
print(x_test.shape, y_test.shape) #(57, 40) (57, 37286)
#130개의 학습 데이터와 57개의 테스트




#검증데이터셋
 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# scaler = StandardScaler()         
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(200, input_dim=34)) 
#model.add(LSTM(32,activation='relu',input_shape = (3,34)))
model.add(Dense(130, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                        filepath = './_ModelCheckPoint/keras27_1_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, callbacks=[es, mcp])



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
acc = model.evaluate(x_test, y_test)[1]
print("loss, acc : ", loss)


y_pred = model.predict(x_test)
print(y_pred[-1])
'''
