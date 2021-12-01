#########################################################
# 각각의 Scaler 특성과 정의 정리할것
#########################################################

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1.데이터

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names) 
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(y)
print(np.unique(y)) #[0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  ##원핫인코딩

print(y)
print(y.shape) #(178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(142, 13) (142, 3)
print(x_test.shape, y_test.shape) #(36, 13) (36, 3)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test) ##y는 타겟이므로 전처리 안함

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=13))
model.add(Dense(10, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

#import time
#start = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('acccuracy: ', loss[1])    


results = model.predict(x_test[:5])
print(y_test[:5])
print(results)


'''
##결과


## 전처리없이
loss :  0.2164507657289505
acccuracy:  0.9166666865348816

<activation = relu>
loss :  0.2709399163722992
acccuracy:  0.8611111044883728


## minmax
loss :  0.0702199712395668
acccuracy:  0.9722222089767456

<activation = relu>
loss :  0.3736262917518616
acccuracy:  0.9722222089767456


## Standard
loss :  0.4314573109149933
acccuracy:  0.9722222089767456

<activation = relu>
loss :  7.284978664756636e-07
acccuracy:  1.0


## Robust
loss :  0.6233945488929749
acccuracy:  0.9722222089767456

<activation = relu>
loss :  0.0002152402448700741
acccuracy:  1.0


## MaxAbs
loss :  0.024375226348638535
acccuracy:  1.0

<activation = relu> 
loss :  0.2461322396993637
acccuracy:  0.9722222089767456
'''