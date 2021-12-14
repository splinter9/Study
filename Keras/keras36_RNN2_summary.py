import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

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
model.add(SimpleRNN(5, input_shape=(3, 1))) ##행은 넣지않는다
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))  ##플랫튼 할필요없이 덴스로 입력가능
model.summary()

'''
35가 되는 이유
'''

# simple_rnn (SimpleRNN)       (None, 5)                 35
# _________________________________________________________________
# dense (Dense)                (None, 80)                480
# _________________________________________________________________
# dense_1 (Dense)              (None, 50)                4050
# _________________________________________________________________
# dense_2 (Dense)              (None, 30)                1530
# _________________________________________________________________
# dense_3 (Dense)              (None, 20)                620
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_5 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 6,936
# Trainable params: 6,936
# Non-trainable params: 0
# _________________________________________________________________


'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=1000)

#4. 평가 예측

model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]]) # y shape (4,)을 (1,3,1)로 바꿔줌
print(result)
'''

'''
from keras.models import Sequential
from keras.layers import SimpleRNN
model = Sequential()
model.add(SimpleRNN(4, input_shape=(2,3)))
# model.add(SimpleRNN(4, input_length=2, input_dim=3))와 동일함.
model.summary()
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn_2 (SimpleRNN)     (None, 4)                 32        
=================================================================
Total params: 32
Trainable params: 32
Non-trainable params: 0
_________________________________________________________________
Total params = recurrent_weights + input_weights + biases

= (num_units*num_units)+(num_features*num_units) + (1*num_units)

= (num_features + num_units)* num_units + num_units

결과적으로,

( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 + unit 개수 ) + ( 1 * unit 개수)

를 참조하여 위의 total params 를 구하면

(4 4) + (3 4) + (1 * 4) = 32 개 임을 알 수 있다.
'''