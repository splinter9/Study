from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout #dropout 텐서플로 레이어에 있는 친구임
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


scaler = MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=30)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1) 
mcp = ModelCheckpoint(monitor= 'val_loss', mode = 'min', verbose =1, save_best_only=True,
                      filepath = './_ModelCheckPoint/keras28_3_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.3, callbacks=[es,mcp])
 
model.save('./_save/keras28_3_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras28_3_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras28_3_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)


'''
====================== 1. 기본출력 ========================
4/4 [==============================] - 0s 665us/step - loss: 0.0317
loss: 0.031682975590229034
r2 스코어: 0.8651320313910204
====================== 2. load_model 출력 ========================
4/4 [==============================] - 0s 665us/step - loss: 0.0317
loss: 0.031682975590229034
r2 스코어: 0.8651320313910204
====================== 3. ModelCheckPoint 출력 ========================
4/4 [==============================] - 0s 630us/step - loss: 0.0264
loss: 0.026385266333818436
r2 스코어: 0.8651320313910204
'''