#########################################################
###########   삼성전자, 키움증권 주가예측 시험   ##########
#########################################################
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1 데이터
path = "D:\\_data\\exam\\"
samsung = pd.read_csv(path + "삼성전자.csv", encoding='cp949', thousands=',', index_col=0, header=0, sep=',')
kium = pd.read_csv(path + "키움증권.csv", encoding='cp949', thousands=',', index_col=0, header=0, sep=',')

# samsung= samsung['시가','고가'].astype('int')
# kium= kium['시가','고가'].astype('int')
print(samsung.dtypes)
print(kium.dtypes)

samsung= samsung.drop(['전일비', 'Unnamed: 6','등락률'], axis=1)
kium= kium.drop(['전일비', 'Unnamed: 6','등락률'], axis=1)

# for i in range(len(samsung.index)):
#     samsung.iloc[i]= int(samsung.iloc[i].replace(',' ,''))
# for i in range(len(kium.index)):
#     for j in range(len(kium.iloc[i])):
#         kium.iloc[i,j] = int(kium.iloc[i,j].replace(',',''))

samsung = samsung.sort_values(['일자'], ascending=[True])
kium = kium.sort_values(['일자'], ascending=[True])
    
samsung=samsung.values
kium=kium.values

np.save('D:\\_data\\exam\\sam.npy', arr=samsung)
np.save('D:\\_data\\exam\\kim.npy', arr=kium)

#print(samsung, kium)
print(samsung.shape, kium.shape) #(1120, 13) (1060, 13)


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column 
        
        if y_end_number > len(dataset): 
            break
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number:y_end_number, 3] 
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1) 
x2, y2 = split_xy5(kium, 5, 1) 
print(x2[0,:], "\n", y2[0])
print(x2.shape)
print(y2.shape)

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=2, test_size = 0.3)

print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)

x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)
print(x2_test.shape)


# #### 데이터 전처리 #####
# from sklearn.preprocessing import StandardScaler
# scaler1 = StandardScaler()
# scaler1.fit(x1_train)
# x1_train_scaled = scaler1.transform(x1_train)
# x1_test_scaled = scaler1.transform(x1_test)
# scaler2 = StandardScaler()
# scaler2.fit(x2_train)
# x2_train_scaled = scaler2.transform(x2_train)
# x2_test_scaled = scaler2.transform(x2_test)
# print(x2_train_scaled[0, :])

# x1_train_scaled = np.reshape(x1_train_scaled,
#     (x1_train_scaled.shape[0], 5, 5))
# x1_test_scaled = np.reshape(x1_test_scaled,
#     (x1_test_scaled.shape[0], 5, 5))
# x2_train_scaled = np.reshape(x2_train_scaled,
#     (x2_train_scaled.shape[0], 5, 5))
# x2_test_scaled = np.reshape(x2_test_scaled,
#     (x2_test_scaled.shape[0], 5, 5))
# print(x2_train_scaled.shape)
# print(x2_test_scaled.shape)

print('================')
print(x1.shape,y1.shape)

from keras.models import Model
from keras.layers import Dense, Input, LSTM

# 모델구성
input1 = Input(shape=(13,))
dense1 = LSTM(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(13,))
dense2 = LSTM(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs=[input1, input2],
              outputs = output3 )


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train, x2_train], y1_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=100, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test, x2_test])

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])
