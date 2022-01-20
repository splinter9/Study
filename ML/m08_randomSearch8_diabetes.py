from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = load_diabetes()
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
#머신러닝에서 모델.스코어는 .프레딕트와 거의 같은 개념

'''
######################################################
# print(model.cv_results_) #훈련에 대한 내용 전체

aaa = pd.DataFrame(model.cv_results_)
print(aaa)

bbb=aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
        #  'split0_test_score','split1_test_score','split2_test_score','split3_test_score','split4_test_score']]

print(bbb)
########################################################

GridSearchCV

# 최적의 메개변수 :  SVC(C=1000, kernel='linear')
# 최적의 파라미터 :  {'C': 1000, 'degree': 3, 'kernel': 'linear'}
# model.score :  1.0   =   accuracy_score :  1.0    둘은같다

# best_score_ :  0.9916666666666668   훈련에서 최고값
# model.score :  0.9666666666666667   val_accuracy 즉 테스트한 최고값
# accuracy_score :  0.9666666666666667  테스트에서 최고값 모델스코어와 동일하다
# 최적튠 ACC 0.9666666666666667


RandomizedSearchCV

Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 메개변수 :  SVC(C=100, gamma=0.001)
최적의 파라미터 :  {'kernel': 'rbf', 'gamma': 0.001, 'C': 100}
best_score_ :  0.9666666666666668
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적튠 ACC 0.9666666666666667
걸린시간: 1.517 초

Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 메개변수 :  SVC(C=1000, degree=4, gamma=0.001, kernel='sigmoid')
최적의 파라미터 :  {'kernel': 'sigmoid', 'gamma': 0.001, 'degree': 4, 'C': 1000}
best_score_ :  0.9916666666666668
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적튠 ACC 0.9666666666666667
걸린시간: 1.499 초

Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 메개변수 :  SVC(C=1, degree=4, kernel='linear')
최적의 파라미터 :  {'kernel': 'linear', 'degree': 4, 'C': 1}
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
최적튠 ACC 0.9666666666666667
걸린시간: 1.492 초



'''


