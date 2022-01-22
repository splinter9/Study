import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sam = pd.read_csv("D:\\_data\\exam\\삼성전자.csv", thousands=',', encoding='CP949')
kim = pd.read_csv("D:\\_data\\exam\\키움증권.csv", thousands=',', encoding='CP949')

print(sam.shape, kim.shape) #(1120, 17) (1060, 17)
sam = sam.drop(range(893,1120), axis=0) #삼성전자 분할 이전 데이터는 잘라낸다
kim = kim.drop(range(893,1060), axis=0) #키움증권도 삼성전자와 행을 맞춰준다 안그러면 앙상블이 안돌아감


x1 = sam.drop(columns=['일자','거래량','고가','종가','저가','시가', '전일비', 'Unnamed: 6', '등락률'], axis=1)
y1 = sam['거래량'] 

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.5, random_state=66)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, test_size=0.5, random_state=66)

print(x1.shape, y1.shape) #(893, 4) (893,)

x2 = kim.drop(columns=['일자','거래량','고가','종가','저가','시가', '전일비', 'Unnamed: 6', '등락률'], axis=1)
y2 = kim['거래량'] 

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.5, random_state=66)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, test_size=0.5, random_state=66)

y1 = np.log1p(y1)
y2 = np.log1p(y2)

print(x2.shape, y2.shape) #(893, 4) (893,)

sam = sam.sort_values(['일자'], ascending=[True])
kim = kim.sort_values(['일자'], ascending=[True])


#삼성전자 모델 input
input1 = Input(shape=(8,))
dense1_1 = Dense(130, activation='relu')(input1)
dense1_2 = Dense(80, activation='relu')(dense1_1)
dense1_3 = Dense(50, activation='relu')(dense1_2)
dense1_4 = Dense(30, activation='relu')(dense1_3)
dense1_5 = Dense(20, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)

#키움증권 모델 input
input2 = Input(shape=(8,))
dense2_1 = Dense(130, activation='relu')(input2)
dense2_2 = Dense(80, activation='relu')(dense2_1)
dense2_3 = Dense(50, activation='relu')(dense2_2)
dense2_4 = Dense(30, activation='relu')(dense2_3)
output2 = Dense(1)(dense2_4)

#앙상블
from tensorflow.keras.layers import Concatenate, concatenate
merge1 = concatenate([output1, output2])

#삼성전자 output
output1_1 = Dense(80, activation='relu')(merge1)
output1_2 = Dense(50, activation='relu')(output1_1)
output1_3 = Dense(30, activation='relu')(output1_2)
output1_4 = Dense(20, activation='relu')(output1_3)
sam_output = Dense(1)(output1_3)

#키움증권 output
output2_1 = Dense(70, activation='relu')(merge1)
output2_2 = Dense(30, activation='relu')(output2_1)
output2_3 = Dense(20, activation='relu')(output2_2)
output2_4 = Dense(10, activation='relu')(output2_3)
kim_output = Dense(1)(output2_4)

model = Model(inputs=[input1, input2], outputs=[sam_output, kim_output])

model.summary()

np.save('D:\\_data\\exam\\sam.npy', arr=sam)
np.save('D:\\_data\\exam\\kim.npy', arr=kim)
model.save('D:\\_data\\exam\\sam_kim_2.h5')

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam',metrics=['mae'])
es = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
model.fit([x1_train, x2_train], [y1_train, y2_train], 
          epochs=1000, batch_size=32, validation_data=([x1_val, x2_val], [y1_val, y2_val]), callbacks=[es])
       
loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("loss : ",loss)
y1_pred,y2_pred=model.predict([x1_test, x2_test])
print('거래량 : ', '삼성전자',y1_test[0], '/ 키움증권',y2_test[0])
print('예상거래량 : ', '삼성전자', y1_pred[-1][-1], '/ 키움증권',y2_pred[-1][-1])

'''
########## 결과 ##########
loss :  [17854067900416.0, 17850997669888.0, 3069551616.0, 3079765.25, 36569.796875]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6051960.5] / 키움증권 [21633.975]

loss :  [14338917138432.0, 14334540382208.0, 4370023424.0, 2688555.25, 44919.1015625]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [8040056.5] / 키움증권 [35823.777]

loss :  [13519291416576.0, 13515015323648.0, 4270683136.0, 2777910.5, 42192.1171875]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6622476.] / 키움증권 [28648.062]

loss :  [16437313798144.0, 16433218060288.0, 4103380992.0, 3035018.5, 40282.42578125]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6229283.5] / 키움증권 [22995.602]

loss :  [19083907039232.0, 19080742436864.0, 3158104064.0, 3118045.25, 37653.671875]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [5781381.] / 키움증권 [24682.46]

loss :  [16942124498944.0, 16938018275328.0, 4114185728.0, 2913606.5, 42440.70703125]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6702113.] / 키움증권 [31097.207]

loss :  [15241749463040.0, 15237964103680.0, 3789652992.0, 2931370.0, 38456.40234375]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6088249.] / 키움증권 [39263.39]

loss :  [14995737804800.0, 14990679474176.0, 5057961472.0, 2869148.5, 44110.703125]
거래량 :  삼성전자 11802494.0 / 키움증권 60487
예상거래량 :  삼성전자 [6979959.5] / 키움증권 [33069.434]



'''

