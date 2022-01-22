import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM

path = "D:\\_data\\exam\\"

sam=pd.read_csv(path+'삼성전자.csv',thousands=',', encoding='CP949')
kim=pd.read_csv(path+'키움증권.csv',thousands=',', encoding='CP949')


sam=sam.sort_values(['일자'], ascending=[True])
kim=kim.sort_values(['일자'], ascending=[True])


x1=sam.drop(range(60,1120), axis=0)
x2=kim.drop(range(60,1060), axis=0)


x1=x1.loc[::-1].reset_index(drop=True)
x2=x2.loc[::-1].reset_index(drop=True)

x1 = x1.drop(columns=['일자','종가', "Unnamed: 6", '전일비' , '등락률','금액(백만)', '신용비' , '개인', '기관','외인(수량)', '외국계' , '프로그램' , '외인비','거래량'], axis=1)
x2 = x2.drop(columns=['일자','종가', "Unnamed: 6", '전일비' , '등락률','금액(백만)', '신용비' , '개인', '기관','외인(수량)', '외국계' , '프로그램' , '외인비', '거래량'], axis=1)
x1 = np.array(x1)
x2 = np.array(x2)
# print(x1.shape, x2.shape)  #(893, 4) (893, 4)

y1 = sam['종가']
y2 = kim['종가']


def split_xy3(dataset, time_steps, y_column):
    x,y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_column-1
       
        if y_end_number>len(dataset):
            break
        tmp_x=dataset[i:x_end_number, :-1]
        tmp_y=dataset[x_end_number-1:y_end_number,-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1,y1=split_xy3(x1,5,2)
x2,y2=split_xy3(x2,5,2)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, train_size=0.8, random_state=66)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(5,2))
dense1=LSTM(50, activation='linear', name='dense1')(input1)
dense2=Dense(30, activation='relu', name='dense2')(dense1)
dense3=Dense(20, activation='relu', name='dense3')(dense2)
output1=Dense(2, activation='linear', name='dense4')(dense3)

input2=Input(shape=(5,2))
dense11=LSTM(50, activation='linear', name='dense11')(input2)
dense12=Dense(30, activation='relu', name='dense12')(dense11)
dense13=Dense(20, activation='relu', name='dense13')(dense12)
dense14=Dense(10, activation='relu', name='dense14')(dense13)
output2=Dense(2, activation='linear', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])

output21 = Dense(70)(merge1)
output22 = Dense(50, activation='linear')(output21)
output23 = Dense(10, activation='relu')(output22)
last_output1 = Dense(1)(output23)

output31 = Dense(70)(merge1)
output32 = Dense(20)(output31)
output33 = Dense(20, activation='linear')(output32)
output34 = Dense(10, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model=Model(inputs=[input1,input2], outputs=[last_output1, last_output2])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])


es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
# mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,save_best_only=True)

start=time.time()
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=100, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es])
end=time.time()-start

print("걸린시간 :" , round(end, 3), '초')

model.save("D:\\_data\\exam\\삼성키움.h5")

loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
y1_pred,y2_pred=model.predict([x1_test, x2_test])


print('예상가 : ', '삼성전자',y1_pred[-1] +1,'원',' / 키움증권',y2_pred[-1]+1,'원')



#예상가 :  삼성전자 [80409.78] 원  / 키움증권 [131140.84] 원
