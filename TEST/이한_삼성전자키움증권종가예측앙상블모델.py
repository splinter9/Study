import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sam = pd.read_csv("D:\\_data\\exam\\삼성전자.csv", thousands=',', encoding='CP949')
kim = pd.read_csv("D:\\_data\\exam\\키움증권.csv", thousands=',', encoding='CP949')

print(sam.shape, kim.shape) #(1120, 17) (1060, 17)
sam = sam.drop(range(893,1120), axis=0) #삼성전자 분할 이전 데이터는 잘라낸다
kim = kim.drop(range(893,1060), axis=0) #키움증권도 삼성전자와 행을 맞춰준다 안그러면 앙상블이 안돌아감


x1 = sam.drop(columns=['일자', '종가', '전일비', '거래량', 'Unnamed: 6', '등락률', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
y1 = sam['종가'] 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.5, random_state=66)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, test_size=0.5, random_state=66)

print(x1.shape, y1.shape) #(893, 4) (893,)

x2 = kim.drop(columns=['일자', '종가', '전일비', '거래량', 'Unnamed: 6', '등락률', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
y2 = kim['종가']

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.5, random_state=66)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, test_size=0.5, random_state=66)

print(x2.shape, y2.shape) #(893, 4) (893,)





#삼성전자 모델 input
input1 = Input(shape=(4,))
dense1_1 = Dense(130, activation='relu')(input1)
dense1_2 = Dense(80, activation='relu')(dense1_1)
dense1_3 = Dense(50, activation='relu')(dense1_2)
output1 = Dense(10, activation='linear')(dense1_3)

#키움증권 모델 input
input2 = Input(shape=(4,))
dense2_1 = Dense(130, activation='relu')(input2)
dense2_2 = Dense(80, activation='relu')(dense2_1)
dense2_3 = Dense(50, activation='relu')(dense2_2)
dense2_4 = Dense(30, activation='relu')(dense2_3)
output2 = Dense(10, activation='linear')(dense2_4)

#앙상블
from tensorflow.keras.layers import Concatenate, concatenate
merge1 = concatenate([output1, output2])

#삼성전자 output
output1_1 = Dense(70, activation='relu')(merge1)
output1_2 = Dense(20)(output1_1)
output1_3 = Dense(10)(output1_2)
sam_output = Dense(1)(output1_3)

#키움증권 output
output2_1 = Dense(70, activation='relu')(merge1)
output2_2 = Dense(20)(output2_1)
output2_3 = Dense(10)(output2_2)
output2_4 = Dense(5)(output2_3)
kim_output = Dense(1)(output2_4)

model = Model(inputs=[input1, input2], outputs=[sam_output, kim_output])

model.summary()


from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam',metrics=['mse'])
es = EarlyStopping(monitor='loss', patience=200, mode='min', verbose=1, restore_best_weights=True)
model.fit([x1_train, x2_train], [y1_train, y2_train], 
          epochs=1000, batch_size=32, validation_data=([x1_val, x2_val], [y1_val, y2_val]), callbacks=[es])
       
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test])
print("loss : ",loss)
y_pred = model.predict([x1_test, x2_test])  
print('종가 : ', '삼성전자',y1_test[0],'원',' / 키움증권',y2_test[0],'원')


### 결과
# loss :  [1758956.5, 198282.625, 1560673.875, 198282.625, 1560673.875]
# 종가 :  삼성전자 78000 원  / 키움증권 109500 원

