#########################################################
##################  DACON WINE   ########################
#########################################################

   
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder



#1 데이터
path = "D:\\_data\\dacon\\wine\\" 
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

y = train['quality']
x = train.drop(['quality'], axis =1) #
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()
le.fit(train['type'])
# x_type = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
x['type'] = le.transform(train['type'])
# print(x)
# y = np.array(y).reshape(-1,1)
# one_hot = OneHotEncoder()
# one_hot.fit(y)
# y = one_hot.transform(y).toarray()
# print(y)



from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

#caler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



#2 모델구성
#        

model = Sequential()
model.add(Dense(100, activation='linear', input_dim=13))
#model.add(Dense(80, activation='linear'))
model.add(Dense(70, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(40, activation='linear'))
#model.add(Dense(30, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()

model.fit(x_train, y_train, epochs = 300, batch_size =1,validation_split=0.3 ,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])



test_flie['type'] = le.transform(test_flie['type'])





##################### 제출용 제작 ####################
result = model.predict(test_flie) + 4
result_recover = np.argmax(result, axis =1).reshape(-1,1)
submission['quality'] = result_recover
#print(submit_file[:10])
submission.to_csv(path + 'LH_WINE_TEST.csv', index=False) # to_csv하면 자동으로 인덱스가 생기게 된다. > 없어져야 함


