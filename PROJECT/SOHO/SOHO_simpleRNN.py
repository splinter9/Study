#########################################################
############### 지역별 SOHO 폐업률 예측  #################
#########################################################  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy

#1. 데이터

# def split_xy(dataset, time_steps, y_column):                
#     x,y = list(), list()                                    

#     for i in range(len(dataset)):                           
#         x_end_number = i + time_steps                       
#         y_end_number = x_end_number + y_column          

#         if y_end_number > len(dataset):                        
#             break

#         tmp_x = dataset[i:x_end_number, :]
#         tmp_y = dataset[x_end_number:y_end_number, -6]
#         x.append(tmp_x)
#         y.append(tmp_y)
        
#     return np.array(x),np.array(y)


path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA_T.csv")

# use_dataset = dataset.drop(['CTPV_NM','STD_YM','BLCK_SP_CD'],axis=1).values        # ,'STD_YM','BLCK_SP_CD'

#print(use_dataset.shape)   (187, 38)

# x,y = split_xy(use_dataset,3,1)

# print(x.shape,y.shape) #(174, 11, 38) (174, 3)



x = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO'],axis=1)
y = dataset['CLSD_CNT']
x, y = x.values, y.values

# print(x.shape) #(187, 40)
# print(y.shape) #(187,)


#le = LabelEncoder()
#y = to_categorical(y)
print(x.shape)      #(187, 40)
print(y.shape)      #(187, 37286)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)  

#데이터표준화
# mean = np.mean(x_train, axis=0)
# std = np.std(x_train, axis=0)

# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std

print(x_train.shape, y_train.shape) #(130, 40) (130, 37286)
print(x_test.shape, y_test.shape) #(57, 40) (57, 37286)
#130개의 학습 데이터와 57개의 테스트




#검증데이터셋
 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

scaler = StandardScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(200, input_dim=34)) 
model.add(Dense(130, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam') #, metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                        filepath = './_ModelCheckPoint/keras27_1_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, callbacks=[es, mcp])



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)


y_pred = model.predict(x_test)
print(y_pred[-1])

