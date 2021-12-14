from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets, metrics
import numpy as np
from tensorflow.python.keras.metrics import accuracy

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

path = "../_data/dacon/heart/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x=train.drop(['id', 'target','thal'], axis=1)
test_file=test_file.drop(['id','thal'], axis=1)
y=train['target']

# print(train.shape) #(151, 15)
#print(train.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#       dtype='object')

# print(test_file.shape) #(152, 14)
# print(test_file.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
#       dtype='object')
# print(submit_file.shape) #(152, 2)
# print(submit_file.columns)
# Index(['id', 'target'], dtype='object')

# y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size =0.8, shuffle=True, random_state = 7)

# print(x_train.shape)  #(120, 13)
# print(x_test.shape)   #(31, 13)

scaler = MinMaxScaler()        
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)


x_train = x_train.reshape(120,3,4,1)
x_test = x_test.reshape(31,3,4,1)
test_file=test_file.reshape(152,3,4,1)
# print(y.shape)   (151,1)

model = Sequential()
model.add(Conv2D(300, kernel_size = (2,2),input_shape = (3,4,1)))                      
model.add(Flatten())
model.add(Dense(122, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=500, mode='min',
                   verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_7_MCP.hdf5')
model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.5, callbacks=[es,mcp])

# model.save('./_save/keras27_7_save_model.h5')

# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

loss = model.evaluate(x_test, y_test)
y_predict=model.predict(x_test)
y_predict=y_predict.round(0).astype(int)
f1=f1_score(y_test, y_predict)
print('loss : ', loss[0])
print('accuracy :  ', loss[1])
# print('f1_score :  ', f1)

results=model.predict(test_file)
results=results.round(0).astype(int)

print('F1_Score :', f1_score(y,results[1:]))

submit_file['target']=results
submit_file.to_csv(path + "MARS.csv", index=False)