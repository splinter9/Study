import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten #대문자클래스 #소문자함수


#1.데이터 
x= np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6]])
y= np.array([4,5,6,7])  #덩어리로 자르는것을 timesteps라고 한다

print(x.shape, y.shape)  ## (4,3) => (4, 3) (4,)

## input_shape = (batch_size, timesteps, feature)
## input_shape = (batch_sizeg 행, timesteps 열, feature 자르는 갯수)
x = x.reshape(4, 3, 1) ## 4행, 3열, 1개씩 자름


#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(130), input_shape=(3, 1), return_sequences=True)) ##행은 넣지않는다
#model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1))) 
##ValueError: Please initialize `Bidirectional` layer with a `Layer` instance. You passed: 50 //레이어에 인스턴스 넣어라
##bidirectional (Bidirectional (None, 20)   240  ##아웃풋이 두배 20개이다

model.add(Conv1D(10,2, input_shape=(3,1)))
model.add(Dense(9, activation='relu'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=100)


#4. 평가 예측
model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]]) # y shape (4,)을 (1,3,1)로 바꿔줌
print(result)

# start = time.time()
# end = time.time() - start
# print('걸린시간:', round(end,3), '초')