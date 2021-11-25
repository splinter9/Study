#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 이 데이터로 훈련해서 최소의 Loss값을 구해보자

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(80))
model.add(Dense(130))                   #히든레이어는 값을 모른다
model.add(Dense(30))                   #하이퍼파라미터튜닝 - 히든레이어값 수정
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss값은 작을수록 좋다, loss에 값을 감축시키는 역할을 해줌(optimizer)
model.fit(x, y, epochs=30, batch_size=1)    #epochs 훈련횟수 
                                            #batch 한번의 batch마다 주어지는 데이터 샘플 size, batch는 나눠진 데이터셋
                                            #interation은 epoch를 나누어서 실행하는 횟수
#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)


##숙제##
#하이퍼파라미튜닝을 해보시오 에포치 50에 값이 4 나올때까지

'''
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(80))
model.add(Dense(130))                   
model.add(Dense(30))                   
model.add(Dense(1))

loss :  0.00017518775712233037
4의 예측값 :  [[4.0032372]]

loss :  0.0008555152453482151
4의 예측값 :  [[4.045392]]

loss :  7.348592043854296e-05
4의 예측값 :  [[3.9824977]]

loss :  0.000462931435322389
4의 예측값 :  [[3.9485955]]

loss :  0.0007943930104374886
4의 예측값 :  [[3.9263585]]

loss :  0.00020359427435323596
4의 예측값 :  [[3.9721153]]

loss :  0.00032198443659581244
4의 예측값 :  [[3.9856043]]

loss :  0.00030879981932230294
4의 예측값 :  [[3.9808867]]

loss :  0.00034728579339571297
4의 예측값 :  [[3.9995875]]

loss :  6.503154145320877e-05
4의 예측값 :  [[3.991743]]

loss :  0.00035291200038045645
4의 예측값 :  [[3.994952]]

loss :  0.0006411660579033196
4의 예측값 :  [[3.9999263]]
'''

