
import numpy as np
from sklearn.datasets import load_wine

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



#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=13))
model.add(Dense(10, activation='linear'))
model.add(Dense(70, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(50, activation='linear'))
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


### 결과 및 정리 ###
'''
loss :  1.073253870010376
acccuracy:  0.4166666567325592
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]
 [0.32406154 0.39190847 0.28403   ]

이 값은 하이퍼튜닝 과정에서 sigmoid를 중간에 넣어서 값이 매우 나쁘게 나옴
3개 값을 출력해야 하는데 임의로 0,1 두개 값으로 중간에 바꿔버려 오염시킨 경우


##sigmoid를 제거하고 하이퍼튜닝한 결과 
 loss :  0.18607741594314575
acccuracy:  0.9444444179534912
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[2.6253960e-03 9.6948035e-02 9.0042657e-01]
 [1.7902829e-01 8.1400734e-01 6.9643566e-03]
 [1.8196541e-04 9.9972385e-01 9.4204202e-05]
 [9.9619663e-01 3.1806622e-03 6.2276266e-04]
 [1.3252998e-03 9.9752277e-01 1.1519121e-03]]
 
아주 양호한 결과를 보임


오늘 수업의 핵심!!

#이진분류(default = sigmoid)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#다중분류(default = y값을 one hot encoding 해줌)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#회귀분석(default = linear)
model.compile(loss='mse', optimizer='adam')
 
'''