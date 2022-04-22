import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. DATA
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,0,0,1]

# [0,0] -> [0] 
# [0,1] -> [0] ... [1,1] ->[1]

# 2. MODEL
# model = LinearSVC()
model = Perceptron()

# 3. FIT
model.fit(x_data, y_data)

# 4. PREDICT
y_predict = model.predict(x_data)
print(x_data, "의 예측결과: ", y_predict)
results = model.score(x_data, y_data)
print(results)
acc = accuracy_score(y_data, y_predict)
print('accuracy_score:', acc)
