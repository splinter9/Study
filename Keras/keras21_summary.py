#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))           # 1 x 5 = 5   +5   +bias로 연산 한번 더해진다
model.add(Dense(3, activation='relu'))     # 5 x 3 = 15  +3
model.add(Dense(4, activation='relu'))     # 3 x 4 = 12  +4           
model.add(Dense(2))                        # 4 x 2 = 8   +2
model.add(Dense(1))                        # 2 x 1 = 2   +1

model.summary()
'''
Total params: 57
Trainable params: 57
Non-trainable params: 0


# 1 x 5 = 5   +5   +bias로 연산 한번 더해진다
# 5 x 3 = 15  +3
# 3 x 4 = 12  +4 
# 4 x 2 = 8   +2
# 2 x 1 = 2   +1
'''


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



