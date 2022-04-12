import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM


path = "D:\\_data\\exam\\"
dataset = pd.read_csv(path + "삼성전자.csv", encoding='cp949', thousands=',')
dataset = dataset.drop(range(893, 1120), axis=0)
x = dataset.drop(['일자','종가',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1)


y = dataset['종가']
print(x,y)
size = 20
def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): 
        subset = data[i : (i+size)]       
        aaa.append(subset)                  
    return np.array(aaa)
x = split_x(x,size)
y = split_x(y,size)
#print(x.columns, x.shape)  # (1060, 13)
  # (1060, 4) (1060,)

# x = x.to_numpy()

# x = x.head(10)
# y = y.head(20)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.7, shuffle=True, random_state = 42)

print(x.shape,y.shape)  # (874, 20, 4) (874, 20)
# x_train = x_train.values.reshape(1120,4,1)
# x_test = x_test.values.reshape(1120,4,1)


# print(x_train.shape, x_test.shape)  # (954, 13) (106, 13)
# print(y_train.shape, y_test.shape)  # (954,) (106,)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# print(x_train.shape, x_test.shape) # (954, 13, 1) (106, 13, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,activation='relu',input_shape = (20,4)))
# model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam', )
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
#                    verbose=1, restore_best_weights=False)
# mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
model.fit(x_train, y_train, epochs=200, batch_size=2,
          validation_split=0.3)

# model.save('../_test/_save/kovt.h5')

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)
print(y_pred[-1])

