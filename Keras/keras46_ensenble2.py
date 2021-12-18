##앙상블 2모델 ==> 2모델

#.1 데이터
import numpy as np
x1 = np.array([range(100),range(301,401)]) # 삼성 저가, 종가 
x2 = np.array([range(101, 201), range(411, 511), range(100,200)]) # 미국국채선물 시가 고가 종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101))  # 타겟, 삼성전자 종가
y2 = np.array(range(101, 201))  # 타겟, 삼성전자 종가


print(x1.shape, x2.shape, y1.shape, y2.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1,x2, y1, y2, train_size=0.8, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape) #(80, 2) (20, 2)
print(x2_train.shape, x2_test.shape) #(80, 3) (20, 3)
print(y1_train.shape, y1_test.shape) #(80,) (20,)
print(y2_train.shape, y2_test.shape) #(80,) (20,)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)


#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

#엮어보자
from tensorflow.keras.layers import concatenate, Concatenate ##단순히이은것,섞은게아님
merge1 = Concatenate()([output1, output2])

#2-3 output모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)


#2-4 output모델2
output31 = Dense(7)(merge1)
output32 = Dense(11)(output31)
output33 = Dense(11, activation='relu')(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)


model = Model(inputs=[input1,input2], outputs=([last_output1,last_output2]))
model.summary()



#3. 컴파일, 훈련 
import time
start = time.time()

model.compile(loss = 'mae', optimizer = 'adam',metrics=['mae'])   
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs = 20)
end = time.time()- start


#4. 평가, 예측 
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
print('loss:', loss)



#loss: [108.29996490478516, 94.18741607666016, 14.112546920776367, 94.18741607666016, 14.112546920776367]
##값이 5개 나오는 이유 [전체로스, y1 로스, y2 로스, y1 MAE , y2 MAE]



'''
앙상블1에서 돌린 훈련// 디버깅요망

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=100, batch_size=1)



#4. 평가예측

mse = model.evaluate([x1_test, x2_test], [y1_train, y2_train], batch_size=1)
# print('mse: ', mse)
y_predict = model.predict([x1_test, x2_test])
loss = model.evaluate([x1_test, x2_test],[y1_train, y2_train])
print('loss : ', loss)
from sklearn.metrics import r2_score
r2 = r2_score([y1_train,y2_train], y_predict) 
print('r2스코어 : ', r2)

for i in range(len(y_predict)):
    print([y1_train,y2_train],[i], y_predict[i])
'''