###############  Cov1D CIFAR100 버전  ################
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 스케일러 적용
scaler = MinMaxScaler()
x_train = x_train.reshape(50000, -1)  # (50000, 3072)
x_test = x_test.reshape(10000, -1)  # (10000, 3072)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# RNN 데이터 모델로 변형
x_train = x_train.reshape(50000, 64, 48)
x_test = x_test.reshape(10000, 64, 48)

#2. 모델구성

model=Sequential()
model.add(Conv1D(64,4, input_shape=(64, 48)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일
import time
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=300, validation_split=0.3, callbacks=[es])
end = time.time() - start

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)
print('훈련시간:', round(end,3), '초')

# start = time.time()
# end = time.time() - start
# print('걸린시간:', round(end,3), '초')