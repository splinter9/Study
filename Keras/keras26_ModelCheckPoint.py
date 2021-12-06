############  모델체크포인트, ModelCheckpoint ##############

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))


# model.summary()
# model.save_weights('./_save/keras25_1_save_weights.h5') > 훈련되기 전에 랜덤값을 넣어서 연산 시켰다


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', verbose = 1) 
                #    restore_best_weights=True)  # 여기서 'verbose'를 하면, Restoring model weights from the end of the best epoch.를 보여준다. 
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', verbose = 1, save_best_only=True,
                      filepath = './_ModelCheckPoint/{epoch}-{val_loss:.2f}.h5')
# 낮은 loss값들을 저장함, checkpoint로 save했을 때에는 매 지점마다 저장을 한다. >> but, 우리가 필요한 것은 마지막
# Epoch 00004: val_loss improved from 106.49902 to 58.51350, saving model to ./_ModelCheckPoint\keras26_1_MCP.hdf5
# Epoch 00009: val_loss improved from 58.51350 to 49.68252, saving model to ./_ModelCheckPoint\keras26_1_MCP.hdf5
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8,
          validation_split=0.3, callbacks = [es,mcp])
end = time.time() - start

print('===================================')
print(hist) 
print('===================================')
print(hist.history) # 딕셔너리 형태로 추출 됨 
print('===================================')
print(hist.history['loss'])
print('===================================')
print(hist.history['val_loss'])
print('===================================')


import matplotlib.pyplot as plt 
plt.figure(figsize=(9,5)) # 판을 깔다.
plt.plot(hist.history['loss'], marker= '.', c='red', label='loss') # 선, 점을 그리다.
plt.plot(hist.history['val_loss'], marker= '.', c='blue', label='val_loss') 
plt.grid() # 격자를 보이게
plt.title('loss') # 제목
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

print('걸린시간:', round(end,3), '초')

model.save('./_save/keras26_1_save_model.h5')  ##
# model.save_weights('./_save/keras25_1_save_weights.h5')
# model.load_weigths('./_save/keras25_3_save_model.h5') 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)



#loss :  18.30216407775879
#r2값은:  0.7810298898523627

#loss :  18.107370376586914
#r2값은:  0.7833604402513994

#loss :  18.30216407775879
#r2값은:  0.7810298898523627