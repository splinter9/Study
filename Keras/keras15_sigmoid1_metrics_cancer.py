
import numpy as np
from sklearn.datasets import load_breast_cancer


#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30) (569,)

#print(y)
#print(y[:10])        ##y값만 찍어봤더니 [0 0 0 0 0 0 0 0 0 0] 이진분류를 써야함
#print(np.unique(y))  ##[0 1] 분류값에서 고유한값, 0과 1밖에 없다는 뜻


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#print(y_test[:11])



#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=30))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(8, activation='linear'))
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

#import time
#start = time.time()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])


#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)    
print(y_test) ## 0 1 만 찍힌다

##loss :  [0.277472585439682, 0.9122806787490845] 
##        [loss='binary_crossentropy',metrics=['accuracy'] 
## 로스값과 매트릭스값은 훈련에 영향을 주지 않은 결과값이다

results = model.predict(x_test[:21])
print(y_test[:21])
print(results)



'''
###결과###
loss :  [0.26316210627555847, 0.9035087823867798]
loss :  [0.432445764541626, 0.859649121761322]
loss :  [0.6534612774848938, 0.640350878238678]
loss :  [0.6535191535949707, 0.640350878238678]
loss :  [0.48254477977752686, 0.8333333134651184]
loss :  [0.34738272428512573, 0.8859649300575256]
loss :  [0.29173731803894043, 0.8947368264198303]
loss :  [0.42565643787384033, 0.8421052694320679]
loss :  [0.6536880731582642, 0.640350878238678]
loss :  [0.2752925455570221, 0.8771929740905762]

'''





'''
### 정리 ###
sigmoid => 이진분류 함수의 계산법으로 로스값은 바이너리크로스엔트로피로 계산한다
accuracy => 정확도
softmax
one hot encoding

'''




'''
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
'''


