#########################################################
# 각각의 Scaler 특성과 정의 정리할것
#########################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_diabetes()
x = datasets.data
y= datasets.target

## minmaxscaler 적용
#print(np.min(x), np.max(x)) #0.0  711.0
#x = x/711.  ##뒤에 점을 찍는 이유는 부동소수점이하 계산이라서 오류발생확률을 낮춘다
#x = x/np.max(x) ##방식은 상동, 데이터, 즉 컬럼 전체를 전처리함 특성구분이 없어 오류남

y = np.log1p(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test) ##y는 타겟이므로 전처리 안함


#2. 모델구성

model = Sequential()
model.add(Dense(180, input_dim=10)) 
model.add(Dense(170,activation='linear'))
model.add(Dense(150,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)



'''
##결과

## 전처리없이
loss :  0.17007100582122803
r2값은:  0.4707175184602108

<activation = relu>
loss :  0.14665977656841278
r2값은:  0.543576234788502


## minmax
loss :  7.5428242683410645
r2값은:  -22.4742278226827

<activation = relu>
loss :  23.194093704223633
r2값은:  -71.1829661451883


## Standard
loss :  50.83774185180664
r2값은:  -157.21351003529165

<activation = relu>
loss :  15.718986511230469
r2값은:  -47.91948483489282


## Robust
loss :  25.531858444213867
r2값은:  -78.45838641948917

<activation = relu>
loss :  2.8671252727508545
r2값은:  -7.922858038523797


## MaxAbs
loss :  4.690659999847412
r2값은:  -13.59792841753371

<activation = relu> 
loss :  10.940450668334961
r2값은:  -33.04807306606009
'''

