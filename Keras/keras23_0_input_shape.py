import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701,801)])
print(x.shape, y.shape) #(3, 100) (2, 100)
x = np.transpose(x)
y = np.transpose(y)
print(x.shape, y.shape) #(100, 3) (100, 1)



#2. 모델구성
from tensorflow.keras.models import Sequential, Model ##모델은 함수형 모델을 의미함
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)


#model = Sequential()
#model.add(Dense(10, input_dim=3))  #(100,3) -> (None, 9)
#model.add(Dense(10, input_shape=(3,))) ##칼럼 갯수를 넣는다 이유는?
#model.add(Dense(9))
#model.add(Dense(8))
#model.add(Dense(1))
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 63
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 8
=================================================================
Total params: 290


Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 63
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 8
=================================================================
Total params: 290
'''





'''
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))           # 1 x 5 = 5   +5   +bias로 연산 한번 더해진다
model.add(Dense(3, activation='relu'))     # 5 x 3 = 15  +3
model.add(Dense(4, activation='relu'))     # 3 x 4 = 12  +4           
model.add(Dense(2))                        # 4 x 2 = 8   +2
model.add(Dense(1))                        # 2 x 1 = 2   +1

model.summary()



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
'''


