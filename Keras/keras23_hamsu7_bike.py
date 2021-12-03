#########################################################
###시퀀스와 함수형 모델을 비교해볼것
#########################################################

import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
# sqrt : 제곱근

#1. 데이터 
path = '../_data/kaggle/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
print(submit_file.columns)    # ['datetime', 'count']


# print(train.info())
# print(test.describe())   
# 'object': 모든 자료형의 최상위형, string형으로 생각하면 된다.   
# 0   datetime    10886 non-null  object는 수치화 할 수 없다. >> 수치화 작업을 해주어야 한다. 
# print(type(train)) # <class 'pandas.core.frame.DataFrame'>
# print(train.describe()) # mean 평균, std 표준편차, min 최소값, 50% 중위값, holiday는 0과 1(휴일), 
# print(train.columns) 
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'], 
# dtype='object')
# print(train.head(3))
# print(train.tail())


x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

# print(x.columns) 

'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
'''   
# print(x.shape)      # (10886, 8)
y = train['count']
# print(y.shape)      # (10886,)
# print(y)

# 로그변환
y = np.log1p(y)

# plt.plot(y)
# plt.show()
# 데이터가 우상향하는 것처럼 한쪽으로 치우친 경우에는 로그 변환 시켜준다. 
# 로그 변환의 가장 큰 문제 : 0이라는 숫자가 나오면 안된다. 
# >> 안나오게 하려면?? 로그하기 전에 1을 더해준다. 


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 36)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) ##y는 타겟이므로 전처리 안함
test_file = scaler.transform(test_file)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model ##모델은 함수형 모델을 의미함
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(8,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

'''
model = Sequential()
model.add(Dense(18, input_dim=8)) 
model.add(Dense(17,activation='linear'))
model.add(Dense(15,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=30, batch_size=10,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)



'''
##결과


## 전처리없이
loss :  1.492845892906189
r2 : 0.25154026844938004
RMSE :  1.221820584119783

<activation = relu>
loss :  1.4248970746994019
r2 : 0.2856069322500131
RMSE :  1.1936907638174834


## minmax
loss :  1.4851773977279663
r2 : 0.25538475537507777
RMSE :  1.2186785842717305

<activation = relu>
loss :  1.9952713251113892
r2 : -0.0003583247392244804
RMSE :  1.4125408393842593


## Standard
loss :  1.495303750038147
r2 : 0.2503077600792547
RMSE :  1.2228261723900622

<activation = relu>
loss :  1.4060481786727905
r2 : 0.29505743718388755
RMSE :  1.1857689781629197


## Robust
loss :  1.4810932874679565
r2 : 0.257432572926524
RMSE :  1.217001644652362

<activation = relu>
loss :  1.3866254091262817
r2 : 0.3047953527963878
RMSE :  1.1775505265443453


## MaxAbs
loss :  1.465261459350586
r2 : 0.2653698983694541
RMSE :  1.210479886519225

<activation = relu> 
loss :  1.460362195968628
r2 : 0.2678261712634713
RMSE :  1.20845454178467
'''