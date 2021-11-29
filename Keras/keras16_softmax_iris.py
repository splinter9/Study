
import numpy as np
from sklearn.datasets import load_iris

#1.데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names) 
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(150, 4) (150,)
#print(y)
##[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
##셔플안하면 원하는 테스트가 안됨 
#print(np.unique(y)) #[0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  ##원핫인코딩

print(y) 
print(y.shape) #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(120, 4) (120, 3)
print(x_test.shape, y_test.shape) #(30, 4) (30, 3)



#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=4))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(8, activation='linear'))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

#import time
#start = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('acccuracy: ', loss[1])    


results = model.predict(x_test[:7])
print(y_test[:7])
print(results)


### 결과 및 정리 ###
'''
loss :  0.05981618911027908
acccuracy:  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[1.14848181e-05 9.99759138e-01 2.29464495e-04]
 [8.14052669e-07 9.94350672e-01 5.64848026e-03]
 [5.79412642e-07 9.87472415e-01 1.25270225e-02]
 [9.99813020e-01 1.87038662e-04 9.06954091e-15]
 [5.30543821e-06 9.99513626e-01 4.81004827e-04]
 [9.06670539e-06 9.99725640e-01 2.65323004e-04]
 [9.99803603e-01 1.96368666e-04 9.73749500e-15]]
'''
