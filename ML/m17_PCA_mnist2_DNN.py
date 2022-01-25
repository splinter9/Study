import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from numpy.core.fromnumeric import cumsum, shape
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
y = np.append(y_train,y_test, axis=0)

print(x.shape)
print(y.shape)

x = x.reshape(70000,-1)

##############################################
# 실습
# PCA를 통해 0.95 이상인 n_components 가 몇개??
# 28*28 이미지를 1차원인 784로 줄세운다
# 0.95
# 0.99
# 0.999
# 1.0
# np.argmax 사용할것
##############################################

pca = PCA(n_components=486)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

# 0이 포함되면 안되니까 1을 더해줌
print(np.argmax(cumsum >=0.95)+1)  # 154
print(np.argmax(cumsum >=0.99)+1)  # 331
print(np.argmax(cumsum >=0.999)+1) # 486
print(np.argmax(cumsum)+1)         # 713

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

#2.모델구성
model = Sequential()
model.add(Dense(64, input_shape=(486, )))   # 최적의 컴포넌트 조합인 154개를 넣는다
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))


import time
start = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)

end = time.time()
loss= model.evaluate(x_test, y_test)
print(loss)


y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(y_pred)
print(y_test)
print('걸린시간 : ', end - start)

'''
DNN 최고값
RNN 최고값
0.95   [0.20554921030998023, 0.9541428685188293]
0.99   [0.17768776416778564, 0.9564285874366076]
0.999  [0.17178104817867280, 0.9563571214675903]
0.1    [0.20638126134872437, 0.9513571262359619]

'''
