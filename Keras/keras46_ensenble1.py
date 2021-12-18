##앙상블 2모델 ==> 1모델

#.1 데이터
import numpy as np
x1 = np.array([range(100),range(301,401)]) # 삼성 저가, 종가
x2 = np.array([range(101, 201), range(411, 511), range(100,200)]) # 미국국채선물 시가 고가 종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001, 1101))  # 타겟, 삼성전자 종가

print(x1.shape, x2.shape, y.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,x2, y, train_size=0.8, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)

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
from tensorflow.keras.layers import concatenate, Concatenate ##단순히엮은것,섞은게아님
merge1 = Concatenate()([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1,input2], outputs=last_output)
model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],y_train,epochs=100, batch_size=1)


#4. 평가예측

mse = model.evaluate([x1_test, x2_test],y_test, batch_size=1)
print('mse: ', mse)
y_predict = model.predict([x1_test, x2_test])
loss = model.evaluate([x1_test, x2_test],y_test)
print('loss : ', loss)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])


#mse:  [0.11607565730810165, 0.11607565730810165]
#loss :  [0.11607686430215836, 0.11607686430215836]
#r2스코어 :  0.9998531316679521

# mse:  [0.5695384740829468, 0.5695384740829468]
# loss :  [0.5695300698280334, 0.5695300698280334]
# r2스코어 :  0.9992793922815078



'''
다른버전
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],y_train,epochs=100, batch_size=1)

resurts = model.eveluate([x1_test, x2_test],y_test)


'''