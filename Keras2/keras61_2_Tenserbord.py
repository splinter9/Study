import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import mnist #우편번호 손글씨
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy



#1. 데이터 가공
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)  >> 흑백
# print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
x_train = x_train.reshape(60000,28,28,1) # reshape는 전체를 다 곱해서 일정하면 상관 없다. (60000,28,14,2)도 가능
x_test = x_test.reshape(10000, 28,28,1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

x = x_train
y = to_categorical(y_train)



#2. 모델
model = Sequential()
model.add(Conv2D(200, kernel_size=(2,2), input_shape=(28, 28, 1)))  # (9, 9, 10) 으로 변환된다 행렬곱 연산이라서 10은 마지막 노드의 갯수, 고정값
model.add(Conv2D(100, (3, 3), activation='relu'))                   # (7, 7,  5) 9-3+1 = 7
model.add(Dropout(0.2))
model.add(Conv2D(100, (2, 2), activation='relu'))                   # (6, 6,  7) 7-2+1 = 5
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))
model.summary()



#3. 컴파일, 훈련
import time
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, SGD, Nadam 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
learning_rate = 0.001 #러닝레이트에 따라 값이 크게 달라진다

optimizer = Adam(learning_rate=learning_rate)
# optimizer = Adadelta(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
# optimizer = Adamax(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
# optimizer = Nadam(learning_rate=learning_rate)
#각 옵티마이저별로 결과값 비교해볼것

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)  #-> 10번 만에 갱신이 안되면 (factor=0.5) LR을 50%로 줄인다
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)


start = time.time()

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=(['acc']))
model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.3, callbacks=[es, reduce_lr, tb])

end = time.time()


#4. 평가

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
print('소요 시간:', end-start)
hist = model.evaluate()
print(hist)


############ 시각화 ################
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc','val_acc'])

plt.show()
