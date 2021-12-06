############  모델체크포인트, ModelCheckpoint ##############

from tensorflow.keras.models import Sequential, Model, load_model 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time

#1. 데이터
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import History
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                 train_size=0.8, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(22, input_dim=13))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일, 훈련


#model.compile(loss='mse', optimizer='adam')
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
#                   restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                      save_best_only=True,
#                      filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5') 

#모델체크포인트 정의와 저장


#start = time.time()
#hist = model.fit(x_train, y_train, epochs=100, batch_size=8, 
          #validation_data=(x_val, y_val))
#          validation_split=0.2, callbacks=[es,mcp])
#end = time.time() - start

#print("===================================================")
#print(hist.history['val_loss'])
#print("===================================================")


#print("걸린시간:" , round(end, 3), '초')

#model.save("./_save/keras26_1_save_model.h5") 

#model.load_weights('/_save/keras25_1_save_weights.h5')
#model.load_weights('/_save/keras25_3_save_weights.h5')
model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')
#model = load_model('./_save/keras26_1_save_model.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)


#loss :  39.00983428955078
#r2값은:  0.5332799064910401

#loss :  39.00983428955078
#r2값은:  0.5332799064910401

#loss :  18.107370376586914
#r2값은:  0.7833604402513994

#loss :  18.30216407775879
#r2값은:  0.7810298898523627

#loss :  18.107370376586914
#r2값은:  0.7833604402513994

loss :  18.107370376586914
r2값은:  0.7833604402513994