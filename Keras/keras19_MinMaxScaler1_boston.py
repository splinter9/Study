#########################################################
# 각각의 Scaler 특성과 정의 정리할것
#########################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_boston()
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
model.add(Dense(180, input_dim=13)) 
model.add(Dense(170,activation='relu'))
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
loss :  0.03578578308224678
r2값은:  0.7471249153749633

<activation = relu>
loss :  0.07347244024276733
r2값은:  0.4808172477899296


## minmax
loss :  0.14103658497333527
r2값은:  0.003384601045860691

<activation = relu>
loss :  0.542452871799469
r2값은:  -2.8331678322128737


## Standard
loss :  0.12495107203722
r2값은:  0.11705048635124993

<activation = relu>
loss :  0.1419192999601364
r2값은:  -0.002853004172965967


## Robust
loss :  0.1405770629644394
r2값은:  0.006631761136999592

<activation = relu>
loss :  0.4455934762954712
r2값은:  -2.148724495281392


## MaxAbs
loss :  0.14096835255622864
r2값은:  0.003866888772745658

<activation = relu> 
loss :  0.504896342754364
r2값은:  -2.567779850982589
'''



