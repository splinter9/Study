#########################################################
# 각각의 Scaler 특성과 정의 정리할것
#########################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_breast_cancer()
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=30))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

#import time
#start = time.time()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)



'''
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)    
print(y_test) ## 0 1 만 찍힌다

##loss :  [0.277472585439682, 0.9122806787490845] 
##        [loss='binary_crossentropy',metrics=['accuracy'] 
## 로스값과 매트릭스값은 훈련에 영향을 주지 않은 결과값이다

results = model.predict(x_test[:21])
print(y_test[:21])
print(results)
'''



'''
##결과


## 전처리없이
loss :  [0.6872871518135071, 0.35672515630722046]
r2값은:  -3.796345266904311e-05

<activation = relu>
loss :  [0.6553786396980286, 0.13450291752815247]
r2값은:  0.1393269116663991


## minmax
loss :  [0.42233240604400635, 0.34502923488616943]
r2값은:  0.9222376945321735

<activation = relu>
loss :  [0.4223664104938507, 0.3391812741756439]
r2값은:  0.9246672577468642


## Standard
loss :  [0.41436460614204407, 0.35087719559669495]
r2값은:  0.9492632667144987

<activation = relu>
loss :  [0.4207583963871002, 0.35087719559669495]
r2값은:  0.9275240892933532


## Robust
loss :  [0.43543151021003723, 0.3391812741756439]
r2값은:  0.8737045747118275

<activation = relu>
loss :  [0.4149298071861267, 0.35087719559669495]
r2값은:  0.9495305871626313


## MaxAbs
loss :  [0.4213559329509735, 0.34502923488616943]
r2값은:  0.9212808317959529

<activation = relu> 
loss :  [0.42047789692878723, 0.34502923488616943]
r2값은:  0.9331241762504248
'''