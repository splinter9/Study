###############  LSMT FETH_COVTYPE 버전  ################
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


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

x=x.reshape(581012,54,1)  #스플릿전에 리쉐이프해야한다

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) #(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape) #(116203, 54) (116203, 8)



#2. 모델구성

model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(54,1)))
model.add(Dense(10, activation='linear'))
model.add(Dense(70, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(8, activation='softmax')) 

##유니크가 7이고 타겟수(y칼럼)가 8개임???


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


