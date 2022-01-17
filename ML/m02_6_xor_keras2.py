import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. DATA
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

# [0,0] -> [0] 
# [0,1] -> [1] ... [1,1] ->[1]

#2. MODEL
# model = LinearSVC()
# model = Perceptron()
# model = SVC() #다항식으로 해결
model = Sequential()
model.add(Dense(130, input_dim=2))
model.add(Dense(80,activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. FIT
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. PREDICT
y_predict = model.predict(x_data)
results = model.evaluate(x_data, y_data)

print(x_data, "의 예측결과: ", y_predict)
print('metrics_acc:' , results[1])

# acc = accuracy_score(y_data, y_predict)
# print('accuracy_score:', acc)
