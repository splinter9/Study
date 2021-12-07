###############  드랍아웃 보스턴 버전  ################

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, Dropout #dropout 텐서플로 레이어에 있는 친구임
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_boston
from tensorflow.python.keras.saving.save import load_model

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)

#2. 모델구성

#드랍아웃은 모델구성에서 적용한다 노드를 가지치기 하는게 아니라 잔가지를 치는 효과가 있다
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dropout(0.2)) # 이전 층 100개 레이어 중에 20% 가지치기 한다는 의미
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80))
model.add(Dropout(0.1)) # 이전 층 100개 레이어 중에 10% 가지치기 한다는 의미
model.add(Dense(50))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

######################################################################
# import datetime
# date = datetime.datetime.now() 
# aaa = date.strftime('%m%d_%H%M')  
# # print(aaa)   # 1206_1644

# filepath = './_ModelCheckPoint/' # : hist의 val_loss
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  #04 = 넷째자리까지 / .4f : 소수점 넷째자리까지
# model_path = ''.join([filepath, 'k26_',aaa,'_', filename])
#             # ./ModelCheckPoint/k26_1206_1644_2500-0.3724.hdf5
######################################################################


es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', verbose = 1, 
                   restore_best_weights=True)  # 여기서 'verbose'를 하면, Restoring model weights from the end of the best epoch.를 보여준다. 
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', verbose = 1, save_best_only=True,
                      filepath = './_ModelCheckPoint/keras28_1_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
          validation_split=0.3, callbacks = [es,mcp])
end = time.time() - start

print('걸린시간:', round(end,3), '초')

model.save('./_save/keras28_1_save_model.h5')  ##
# model.save_weights('./_save/keras25_1_save_weights.h5')
# model.load_weigths('./_save/keras25_3_save_model.h5') 
# model = load_model('./_ModelCheckPoint/keras26_1_MCP.hDF5')


#4. 평가, 예측

print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

print ('====================== 2. load_model 출력 ========================')
model2 = load_model('./_save/keras28_1_save_model.h5')

loss2 = model2.evaluate(x_test, y_test)
print('loss:', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어:', r2)

print ('====================== 3. ModelCheckPoint 출력 ========================')
model3 = load_model('./_ModelCheckPoint/keras28_1_MCP.hdf5')

loss3 = model3.evaluate(x_test, y_test)
print('loss:', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



'''
====================== 1. 기본출력 ========================
4/4 [==============================] - 0s 664us/step - loss: 43.2750
loss: 43.27497100830078
r2 스코어: 0.33260172707339464
====================== 2. load_model 출력 ========================
4/4 [==============================] - 0s 665us/step - loss: 43.2750
loss: 43.27497100830078
r2 스코어: 0.33260172707339464
====================== 3. ModelCheckPoint 출력 ========================
4/4 [==============================] - 0s 538us/step - loss: 43.2750
loss: 43.27497100830078
r2 스코어: 0.33260172707339464
'''
