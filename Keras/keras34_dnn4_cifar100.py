
import numpy as np 
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 가공
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)       # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)         # (10000, 32, 32, 3) (10000, 1)
x_train = x_train.reshape(50000,32,32,3)  # reshape는 전체를 다 곱해서 일정하면 상관 없다. (60000,28,14,2)도 가능
x_test = x_test.reshape(10000, 32,32,3)
print(np.unique(y_train, return_counts=True)) # 0~99 백개

x= x_train
y_train = to_categorical(y_train)   
y_test = to_categorical(y_test)

x_train= x_train.reshape(50000,-1)       # 4차원 (50000,32,32,3)을 가로로 1자로 쫙펴준다.  행 세로 열 가로 
x_test = x_test.reshape(10000,-1)        # -1이라는 의미는 1열 이후의 모든 차원을 한줄로 만들었다는 약속

scaler= StandardScaler()
scaler.fit(x_train)                      #2차원으로 바꾼 데이터를 스케일 적용한다
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)


#2.모델구성
model = Sequential()
#model.add(Dense(64, input_shape(32*32*3, ))) #1차원 형태로 50000을 제외한 32*32*3 의 매트릭스가 평평하게 되어서 들어옴
model.add(Dense(64, input_shape=(3072, )))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.3, callbacks=[es]) 


loss= model.evaluate(x_test, y_test)
print('loss: ', loss)


#### 결과
#loss:  [3.567258596420288, 0.17059999704360962]
#loss:  [3.5826497077941895, 0.15880000591278076]
#
