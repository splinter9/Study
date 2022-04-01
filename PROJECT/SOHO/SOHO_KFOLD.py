from sklearn.model_selection import cross_validate, KFold
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split



path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA_h.csv")
num_folds = 2


x = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO','MED_ARR_AMT','CLSD_CNT','CLSD_RATIO'],axis=1)
y = dataset['CLSD_RATIO']


print(x.shape)      #(187, 40)
print(y.shape)      #(187, 37286)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)  

kfold = KFold(n_splits=num_folds, shuffle=True)

for train, test in kfold.split(x, y):
    
  # Define the model architecture
    model = Sequential()
    model.add(Dense(200, input_dim=32)) 
    model.add(Dense(180, activation='relu'))
    model.add(Dense(130, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

model.compile(loss='mae', optimizer='adam')
history = model.fit(x_train, y_train, batch_size=1, epochs=500)
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores)
