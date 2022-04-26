import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def split_xy(dataset, time_steps, y_column):                
    x,y = list(), list()                                    

    for i in range(len(dataset)):                           
        x_end_number = i + time_steps                       
        y_end_number = x_end_number + y_column             

        if y_end_number > len(dataset):                        
            break

        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, -6]
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)

path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA_T.csv")

use_dataset = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO'],axis=1).values 

# print(use_dataset.shape)   #(187, 38)
print(use_dataset)

x,y = split_xy(use_dataset,3,1)
print(y.shape)



print(x.shape,y.shape) #(174, 11, 38) (174, 3)

le = LabelEncoder()
y = to_categorical(y)
print(x.shape)      #(174, 11, 38)
print(y.shape)      #(174, 3, 37286)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)  

print(x_train.shape, y_train.shape) #(121, 11, 38) (121, 3, 37286)
print(x_test.shape, y_test.shape) #(53, 11, 38) (53, 3, 37286)

# 데이터표준화
# mean = np.mean(x_train, axis=0)
# std = np.sdt(x_train, axis=0)

# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std

# 검증데이터셋
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# scaler = StandardScaler()         
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(200, input_dim=38)) 
#model.add(LSTM(32,activation='relu',input_shape = (11,38)))
model.add(Dense(130, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = './_ModelCheckPoint/keras27_1_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, callbacks=[es])


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
acc = model.evaluate(x_test, y_test)[1]
print("loss, acc : ", loss)


y_pred = model.predict(x_test)
print(y_pred[-1])
