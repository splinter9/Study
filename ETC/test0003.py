import numpy as np
from sklearn.datasets import fetch_covtype

#1.데이터

datasets = fetch_covtype()
#print(datasets.DESCR)
#print(datasets.feature_names) 
#['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(y) #[5 5 2 ... 3 3 3]
#print(np.unique(y)) #[1 2 3 4 5 6 7]


##케라스 버전
#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)  ##원핫인코딩
#print(y.shape) #(581012, 8)

##사이킷런버전
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(sparse=False)
#y = ohe.fit_transform(y.reshape(-1,1))

##판다스버전
import pandas as pd
pd.get_dummies(y)
#pd.get_dummies(y, drop_first=True)

print(y.shape)
print(np.unique(y))