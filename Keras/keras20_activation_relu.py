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
model.add(Dense(8, activation='relu')) #activation을 바꿔가면서 튠 해볼수있다
model.add(Dense(13, activation='relu'))                   
model.add(Dense(3, activation=''))                   
model.add(Dense(1)) #디폴트는 라이너이다

##히든레이어는 값을 모른다
##하이퍼파라미터튜닝 - 히든레이어값 수정

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



