from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)



#2.MODEL

model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True) 
#Fitting 5 folds for each of 42 candidates, totalling 210 fits


#3.FIT
model.fit(x_train, y_train)

#4. COMPILE

# x_test = x_train
# y_test = y_train

# print("최적의 메개변수 : ", model.best_estimator_)
# print("최적의 파라미터 : ", model.best_params_)

# print("best_score_ : ", model.best_score_)
# print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적튠 ACC",accuracy_score(y_test, y_pred_best))

#머신러닝에서 모델.스코어는 .프레딕트와 거의 같은 개념





# 최적의 메개변수 :  SVC(C=1000, kernel='linear')
# 최적의 파라미터 :  {'C': 1000, 'degree': 3, 'kernel': 'linear'}
# model.score :  1.0   =   accuracy_score :  1.0    둘은같다

# best_score_ :  0.9916666666666668   훈련에서 최고값
# model.score :  0.9666666666666667   val_accuracy 즉 테스트한 최고값
# accuracy_score :  0.9666666666666667  테스트에서 최고값 모델스코어와 동일하다
# 최적튠 ACC 0.9666666666666667



