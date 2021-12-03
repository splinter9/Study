#########################################################
###시퀀스와 함수형 모델을 비교해볼것
#########################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_iris()
x = datasets.data
y= datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

## minmaxscaler 적용
#print(np.min(x), np.max(x)) #0.0  711.0
#x = x/711.  ##뒤에 점을 찍는 이유는 부동소수점이하 계산이라서 오류발생확률을 낮춘다
#x = x/np.max(x) ##방식은 상동, 데이터, 즉 컬럼 전체를 전처리함 특성구분이 없어 오류남

#y = np.log1p(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test) ##y는 타겟이므로 전처리 안함

#2. 모델구성
from tensorflow.keras.models import Sequential, Model ##모델은 함수형 모델을 의미함
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

'''
model = Sequential()
model.add(Dense(10, activation='linear', input_dim=4))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''
#3. 컴파일, 훈련

#import time
#start = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('acccuracy: ', loss[1])    


results = model.predict(x_test[:7])
print(y_test[:7])
print(results)


'''
##결과


## 전처리없이
loss :  0.05885813385248184
acccuracy:  0.9777777791023254

<activation = relu>
loss :  0.0767492800951004
acccuracy:  0.9777777791023254


## minmax
loss :  0.09358493238687515
acccuracy:  0.9555555582046509

<activation = relu>
loss :  0.0907740369439125
acccuracy:  0.9555555582046509


## Standard
loss :  0.1134229451417923
acccuracy:  0.9555555582046509

<activation = relu>
loss :  0.08700086921453476
acccuracy:  0.9777777791023254


## Robust
loss :  0.07952430844306946
acccuracy:  0.9555555582046509

<activation = relu>
loss :  0.18990902602672577
acccuracy:  0.9333333373069763


## MaxAbs
loss :  0.07491552829742432
acccuracy:  0.9555555582046509

<activation = relu> 
loss :  0.08981132507324219
acccuracy:  0.9555555582046509
'''