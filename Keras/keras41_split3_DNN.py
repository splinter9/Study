import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
import time

a = np.array(range(1,101))                     ## [1~10]
x_predict = np.array(range(96, 106))           ##96부터 106까지
y = np.array(range(1,106))

size = 5    # x 4개, y 1개                                  ## size는 5 라는 숫자로 정한다

def split_x(dataset, size):                     ## split_x는 dataset부터 size 크기로 배열한다
    aaa = []                                    ## aaa는 리스트다
    for i in range(len(dataset) - size + 1):    ## for i in range 반복한다 dataset갯수에 뺀다 (size크기에 +1)해서 사이즈 잘라서 다음줄에 하나 더해서 시작
        subset = dataset[i : (i + size)]        ## subset은 dataset을 리스트로 만드는데 for 시작부너 시작에 사이즈를 더해준값을 잘라낸다
        aaa.append(subset)                      ## aaa리스트에 돌아간 dataset을 추가해준다
    return np.array(aaa)                        ## 이렇게 만들어진 aaa리스트를 넌파이를 적용한다

bbb = split_x(a, size)                          ## bbb는 split_x를 a부터 size 크기로 배열한다
print(bbb)                                      ## bbb를 출력한다
print(bbb.shape)

x = bbb[:, :4]
y = bbb[:, 4]
print(x, y)
print(x.shape, y.shape)  #(96, 4) (96,)

ccc = split_x(x_predict, 5)
#print(ccc)

x_predict = ccc[:, :4]

#x_predict = x_predict.reshape(6,4,1)
#x = x.reshape(96,4,1)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(4,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
start = time.time()
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs=100)
end = time.time() - start


#4. 평가 예측
model.evaluate(x, y)
result = model.predict(x_predict) 


print(result)
print('걸린시간:', round(end,3), '초')
