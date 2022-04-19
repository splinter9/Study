
import numpy as np
from sklearn.datasets import load_iris

# 1.데이터

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
# y = to_categorical(y)  ##원핫인코딩

print(y) 
print(y.shape) #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(120, 4) (120, 3)
print(x_test.shape, y_test.shape) #(30, 4) (30, 3)



# 2.모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(10, activation='linear', input_dim=4))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(5))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC() #기본구성 ()안에 


# 3.컴파일, 훈련

#import time
#start = time.time()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

model.fit(x_train, y_train)

#end = time.time() - start
#print("걸린시간:" , round(end, 3), '초')


# 4. 평가, 예측
# loss = model.evaluate(x_test, y_test) 
# print('loss : ', loss[0])
# print('acccuracy: ', loss[1])    
result = model.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)



print(result) #모델이 알아서 분류라서 acc 값으로 나온다
print(acc)

# 0.9666666666666667
# 0.9666666666666667
