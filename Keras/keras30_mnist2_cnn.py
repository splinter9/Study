import numpy as np 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import mnist #우편번호 손글씨
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터 가공
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  >> 흑백
# print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
x_train = x_train.reshape(60000,28,28,1) # reshape는 전체를 다 곱해서 일정하면 상관 없다. (60000,28,14,2)도 가능
x_test = x_test.reshape(10000, 28,28,1)
# print(x_train.shape)
# print(np.unique(y_train, return_counts=True))  # (60000, 28, 28, 1)(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

'''
print(x_train[0])
print('y_train[0]번째 값 : ', y_train[0])
import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')
plt.show()
'''

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) # (60000, 10)


model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1)))  # (9, 9, 10) 으로 변환된다 행렬곱 연산이라서 10은 마지막 노드의 갯수, 고정값
model.add(Conv2D(5, (3, 3), activation='relu'))                    # (7, 7,  5) 9-3+1 = 7
model.add(Dropout(0.2))
model.add(Conv2D(10, (2, 2), activation='relu'))                    # (6, 6,  7) 7-2+1 = 5
model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Dense(10, activation='softmax'))
model.summary()


'''
테스트데이터는 건들지말고
아웃풋10개
평가지표 acc 0.98 이상
val_스플릿
테스트를 어큐러시와 로스로 평가한다

'''
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3, callbacks=[es,mcp])

model.save('./_save/keras30_2_save_model.h5')

print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



'''
scaler = MaxAbsScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
'''


################### CNN 주요 내용 정리 #########################
'''
model.add(Conv2D(a, kernel_size = (b,c), input_shape = (q,w,e))) 
1) a = 출력 채널 
2) b,c = 필터, 커넬 사이즈와 같은 말이며, 이미지의 특징을 찾아내기 위한
공용 파라미터이다. 
3) e : 입력 채널, RGB 
- 이미지 픽셀 하나하나는 실수이며, 컬러사진을 표현하기 위해서는 RGB 3개의
실수로 표현해야 한다. 
'''