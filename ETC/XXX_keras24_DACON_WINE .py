#########################################################
##################  DACON WINE 소담버전 #################
#########################################################  
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
#1 데이터
path = "D:\\_data\\dacon\\wine\\" 
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

y = train['quality']
x = train.drop(['id','quality'], axis =1) #

le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!
#--to_categorical은 빈부분을 채우니 주의 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#-------------------------
y = np.array(y).reshape(-1,1)
enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
enc.fit(y)
y = enc.transform(y).toarray()

# print(y.shape)


# print(np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#2 모델구성#        
node_num = [100, 80, 60, 40, 60, 40, 20, 30, 20, 10, 40, 30, 20, 10, 5, 2]
# node_num = [226, 172, 129, 76, 65, 52, 42, 33, 26, 17, 9, 8, 5, 3, 2]
model = Sequential()
model.add(Dense(node_num[0],activation ='relu', input_dim = x_train.shape[1])) #activation = 'linear'
model.add(Dense(node_num[1],activation ='sigmoid' ))
model.add(Dense(node_num[2],activation ='relu'))
model.add(Dense(node_num[3],activation ='relu' )) 
model.add(Dense(node_num[4],activation ='relu')) 
model.add(Dense(node_num[5],activation ='sigmoid'))
model.add(Dense(node_num[6],activation ='relu' )) #
model.add(Dense(node_num[7],activation ='relu')) #
model.add(Dense(node_num[8],activation ='relu' )) 
# model.add(Dense(node_num[9],activation ='relu' )) 
model.add(Dense(node_num[10],activation ='relu'))
# model.add(Dense(node_num[11],activation ='relu' )) 
# model.add(Dense(node_num[12],activation ='relu' )) 
model.add(Dense(node_num[13],activation ='relu')) 
# model.add(Dense(node_num[14])) 
# model.add(Dense(node_num[15])) 
model.add(Dense(y.shape[1], activation = 'softmax'))
#-------> Acc 0.61248182325
#3. 컴파일, 훈련
epoch = 10000
opt="Adamax"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 100
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'min', restore_best_weights=True)
'''
verbos - 
0  없다
1  모두 보여짐
2  loss
3~ epoch만
'''
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =2,verbose = 2,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
# print("epochs :",epoch)


test_flie['type'] = le.transform(test_flie['type'])
test_flie = test_flie.drop(['id'], axis =1) #
test_flie = scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
# print(result[:5])

result_recover = np.argmax(result, axis = 1).reshape(-1,1) + 4
# print(result_recover[:5])
print(np.unique(result_recover)) # np.unique()

submission['quality'] = result_recover

# # print(submission[:10])

# print(result_recover)

acc_list = hist.history['accuracy']
acc = opt + "_acc_"+str(acc_list[-patience_num]).replace(".", "_")
print(acc)
# acc= str(loss[1]).replace(".", "_")
model.save(f"./_save/keras24_dacon_save_model_{acc}.h5")
submission.to_csv(path+f"sampleHR_{acc}.csv", index = False)
'''
accuracy :  0.5672333836555481
'''

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()