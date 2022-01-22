import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ss = pd.read_csv("D:/_data/exam/삼성전자.csv", thousands=',', encoding='CP949')
ss = ss.drop(range(20, 1120), axis=0)
ki = pd.read_csv("D:/_data/exam/키움증권.csv", thousands=',', encoding='CP949')
ki = ki.drop(range(20, 1060), axis=0)

# 인덱스재배열
ss = ss.loc[::-1].reset_index(drop=True)
ki = ki.loc[::-1].reset_index(drop=True)
x_ss = ss.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)','신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ss = np.array(x_ss)
x_ki = ki.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)','신용비',  '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ki = np.array(x_ki)

# split 함수 정의
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column-1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 1:]
        tmp_y = dataset[x_end_number-1:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_ssp, y_ssp = split_xy(x_ss, 2, 4)
x_kip, y_kip = split_xy(x_ki, 2, 4)


x1_train, x1_test, y1_train, y1_test = train_test_split(x_ssp, y_ssp, train_size=0.8, random_state=66)
x2_train, x2_test, y2_train, y2_test = train_test_split(x_kip, y_kip, train_size=0.8, random_state=66)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#삼성 input
input1 = Input(shape=(2, 3))
dense1_1 = LSTM(32, activation='relu')(input1)
dense1_2 = Dense(16, activation='relu')(dense1_1)
output1 = Dense(4, activation='linear')(dense1_2)

#키움 input
input2 = Input(shape=(2, 3))
dense2_1 = LSTM(32, activation='relu')(input2)
dense2_2 = Dense(16, activation='relu')(dense2_1)
output2 = Dense(4, activation='linear')(dense2_2)

#앙상블
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])

#삼성 out
output1_1 = Dense(16, activation='relu')(merge1)
output1_2 = Dense(8)(output1_1)
ss_output = Dense(1)(output1_2)

#키움 out
output2_1 = Dense(16, activation='relu')(merge1)
output2_2 = Dense(8)(output2_1)
ku_output = Dense(1)(output2_2)


model = Model(inputs=[input1, input2], outputs=[ss_output, ku_output])


model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=200, restore_best_weights=True)
model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=1000, batch_size=1, validation_split=0.3, callbacks=[es])

#model.save_weights("D:\\_data\\exam\\삼성키움_시가.h5")  
model.load_weights("D:\\_data\\exam\\삼성키움_시가[77770.11] [108161.48].h5")

loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ', loss)
ss_pred, ki_pred = model.predict([x1_test, x2_test])
print('예측값 : ', ss_pred[-1], ki_pred[-1])

