import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
# sqrt : 제곱근

#1. 데이터 
path = './_data/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
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
        train_size =0.9, shuffle=True, random_state = 36)

#2. 모델구성

model = Sequential()
model.add(Dense(180, input_dim=8)) 
model.add(Dense(170,activation='linear'))
model.add(Dense(160))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1,
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
################################
로그 변환 전
loss :  24695.318359375
r2 : 0.26067632198517987
RMSE :  157.14742020627182

로그 변환 적용 후
loss :  1.4911260604858398
r2 : 0.25609501552210445
RMSE :  1.221116705839553
##################################
'''


##################### 제출용 제작 ####################
results = model.predict(test_file)

submit_file ['count'] = results

print(submit_file[:10])

submit_file.to_csv(path + 'LH_BIKE_TEST.csv', index=False) # to_csv하면 자동으로 인덱스가 생기게 된다. > 없어져야 함


'''
파일 생성 결과
              datetime     count
0  2011-01-20 00:00:00  3.799795
1  2011-01-20 01:00:00  3.756129
2  2011-01-20 02:00:00  3.756129
3  2011-01-20 03:00:00  3.781478
4  2011-01-20 04:00:00  3.781478
5  2011-01-20 05:00:00  3.631876
6  2011-01-20 06:00:00  3.589456
7  2011-01-20 07:00:00  3.714853
8  2011-01-20 08:00:00  3.733255
9  2011-01-20 09:00:00  3.923525
'''