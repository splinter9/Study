#배치사이즈를 없애고 돌려봐라 그리고 배치사이즈 디폴트값을 찾아봐라

import numpy as np
from sklearn.datasets import fetch_covtype

#1.데이터

datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names) 
#['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,)
print(y) #[5 5 2 ... 3 3 3]
print(np.unique(y)) #[1 2 3 4 5 6 7]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  ##원핫인코딩
print(y.shape) #(581012, 8)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape) #(116203, 54) (116203, 8)



#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=54))
model.add(Dense(10, activation='linear'))
model.add(Dense(70, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(8, activation='softmax')) 

##유니크가 7이고 타겟수(y칼럼)가 8개임


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('acccuracy: ', loss[1])    


results = model.predict(x_test[:5])
print(y_test[:5])
print(results)



### 결과 및 정리 ###
'''
loss :  0.6505149006843567
acccuracy:  0.7181226015090942
[[0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]]
[[1.3477453e-14 7.2411144e-01 2.6227605e-01 7.1486511e-08 8.8144367e-11
  7.6313934e-04 7.2849787e-07 1.2848499e-02]
 [5.1537650e-12 5.7727475e-02 9.3716645e-01 1.1878162e-04 6.5105676e-04
  2.0010993e-03 2.3261867e-03 8.8955130e-06]
 [4.9285138e-12 8.0698484e-01 1.7660528e-01 1.7762960e-07 3.5280650e-09
  1.0693371e-03 1.4682388e-05 1.5325610e-02]
 [2.4152165e-11 7.2045989e-02 9.1990042e-01 1.4621082e-04 7.8683061e-04
  4.8304731e-03 2.2620989e-03 2.8041144e-05]
 [5.1767726e-13 5.0561506e-01 4.8011741e-01 2.4212384e-05 1.7939278e-09
  2.7682295e-03 5.5456851e-05 1.1419683e-02]]



### 배치 사이즈 비교 ###
batch_size=1로 지정후 실행한 경우
371847/371847 [==============================] - 218s 587us/step - loss: 1.0092 - accuracy: 0.6511 - val_loss: 0.7717 - val_accuracy: 0.6701

batch_size를 넣지않고 디폴트값으로 실행한 경우
11621/11621 [==============================] - 7s 618us/step - loss: 0.6715 - accuracy: 0.7081 - val_loss: 0.6590 - val_accuracy: 0.7160

371847, 11621 로 그 비율이 약 31.9978배임을 알수있습니다.
그러므로 배치사이즈의 디폴트값은 약 '32'입니다

'''