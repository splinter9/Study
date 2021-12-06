import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x = train.drop(columns=['id', 'quality'], axis=1)
y = train['quality']
test_file = test_file.drop(columns=['id'], axis=1)

y = pd.get_dummies(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

'''
label encoding을 통해서 str로 되있는 정보들을 숫자값으로 변환시켜준다.(문자형은 계산을 못함)
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

# print(np.unique(y)) # [4, 5, 6, 7, 8]
# print(x.shape, y.shape) # (3231, 12) (3231,)

model = Sequential()
model.add(Dense(100, input_dim=x.shape[1]))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True)
model.fit(x_train, y_train, epochs=10000, validation_split=0.2, verbose=3, batch_size=10, callbacks=[es,mcp])




loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

acc = str(round(loss[1], 4))
model.save("./_save/dacon_wine_{}.h5".format(acc))

############################## 제출용 #####################################

result = model.predict(test_file)
'''
argmax 함수를 사용하여 다중분류로 예측된 값들중 가장 높은 값을 선택하여준다.
'''
result_recover = np.argmax(result, axis=1).reshape(-1, 1) +4  # +4를 해줌으로 quality의 기준값들에 
submit_file['quality'] = result_recover
# print(result_recover[:20])
# print(np.unique(result_recover))
submit_file.to_csv(path + "winequality.csv", index=False) 