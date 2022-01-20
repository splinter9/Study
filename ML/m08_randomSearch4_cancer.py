import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5,6]},                               # 12
    {"C":[1, 10, 100, 1000, 10000], "kernel":["rbf"],"gamma":[0.001, 0.0001, 0.00001]},              # 6
    {"C":[1, 10, 100, 1000],"kernel":["sigmoid"],"gamma":[0.01, 0.001, 0.0001], "degree":[3,4,5,6]}  # 
    ]   # 총42개


#2.MODEL

#model = SVC(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66, n_iter=20) 
#model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) 

#Fitting 5 folds for each of 42 candidates, totalling 210 fits


#3.FIT
import time
start = time.time()
model.fit(x_train, y_train)

#4. COMPILE

# x_test = x_train
# y_test = y_train

print("최적의 메개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC",accuracy_score(y_test, y_pred_best))

end = time.time() - start
print('걸린시간:', round(end,3), '초')

'''
Fitting 5 folds for each of 20 candidates, totalling 100 fits
최적의 메개변수 :  SVC(C=100, gamma=1e-05)
최적의 파라미터 :  {'kernel': 'rbf', 'gamma': 1e-05, 'C': 100}
best_score_ :  0.9472527472527472
model.score :  0.9385964912280702
accuracy_score :  0.9385964912280702
최적튠 ACC 0.9385964912280702
걸린시간: 15.033 초
'''
