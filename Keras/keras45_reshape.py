from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D ,Flatten, Dropout, Reshape, Conv1D, LSTM
from tensorflow.keras.datasets import mnist


model = Sequential()

#model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1)))  # (4, 4, 10) 으로 변환된다 행렬곱 연산이라서 10은 마지막 노드의 갯수, 고정값
#model.add(Conv2D(5, (2,2), activation='relu'))                     # (3, 3, 5)
#model.add(Conv2D(7, (2,2), activation='relu'))                     # (2, 2, 7)
#model.add(Flatten())
#model.summary()

#input_shape = (a,b,c) > kernel_size = (d,e) = (a-d+1, b-e+1)  // 

model.add(Conv2D(10, kernel_size=(2,2), strides=1,
                 padding='same', input_shape=(28, 28, 1)))  # (a, b, 10) 으로 변환된다 행렬곱 연산이라서 10은 마지막 노드의 갯수, 고정값

model.add(MaxPooling2D()) #디폴트는 2, 괄호안에 3을 넣으면 1/3이 줄어든다
model.add(Conv2D(5, (2, 2), activation='relu'))       # 13, 13, 5
model.add(Conv2D(7, (2, 2), activation='relu'))       # 12, 12, 7
model.add(Conv2D(7, (2, 2), activation='relu'))       # 11, 11, 7
model.add(Conv2D(10, (2, 2), activation='relu'))      # 10, 10, 10
model.add(Flatten())                                  # (n, 1000)
model.add(Reshape(target_shape=(100,10)))             #(None, 100, 10) 플랫튼한 차원을 다시 되돌려준다
model.add(Conv1D(5, 2))                               #(N, 99, 5)
model.add(LSTM(15))
model.add(Dense(5, activation='softmax'))
                 
# (n, 100, 10)
#model.add(Dense(32))
#model.add(Dropout(0.2))
#model.add(Dense(16))
#model.add(Dense(5, activation='softmax'))
model.summary()

##padding=  sama 같게, valid 둘러쌈
##strides= 1, 2, 3, 4, 숫자 만큼 칸을 건너가서 시작

## 5X5 크기의 흑백(1), 컬러는(3)
## 10은 다음레이어로 전달하는 노드 갯수 
## 커널사이즈는 자르는 크기 (2,2) 이건 임의로 자르는거임 크게 자를수록 연산이 많다 
## 이미지는 3차원 이상(가로x세로x색깔x행렬)데이터이기에 innput_shape를 쓴다

# 반장, 이한, 예람, 명재, 모나리자 -> 
# LabelEncoder
# 0, 1, 2, 3, 4  -> (5,)     -> (5,1)
               # [0,1,2,3,4]    [[0],[1],[2],[3],[4]]
# 데이터 수집과 정제를 할 때 모든 데이터의 shape(모양)가 동일해야한다. 
# input_shape = (5,5,1))) 마지막 1 생략 불가함 >> 무조건 1 혹은 3을 기입
# input_shape = (5,5,1))) >> (4,4,10)# 데이터 수집과 정제를 할 때 모든 데이터의 shape(모양)가 동일해야한다. 
# input_shape = (5,5,1))) 마지막 1 생략 불가함 >> 무조건 1 혹은 3을 기입
# input_shape = (5,5,1))) >> (4,4,10)
#4차원은 어떻게 2차원으로 바꾸지? reshape? 행렬을 1열 리스트로 나열해준다
# (n, 6, 6, 3) => (n, 6 X 6 X 3) => (n, 253)



'''
##과제###

Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50           
_________________________________________________________________      
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 7)           147
=================================================================
Total params: 402
Trainable params: 402
Non-trainable params: 0
_________________________________________________________________

1> Param # 은 왜 50인가? 
2> (None, 4, 4, 10)

1>> model.add(Conv2D(c, kernel_size = (a,b) , input_shape = (10,10,1)))
 (필터 크기 axb) x  (입력 채널(RGB)) x (출력 채널 c) + (출력 채널 c bias)
두 번째 연산은
위 레이어의 출력채널 c를 아래의 출력 채널값과 곱하여 계산한다. 
model.add(Conv2D(필터, kernel_size = (a,b) , input_shape = (10,10,채널)))

'''
