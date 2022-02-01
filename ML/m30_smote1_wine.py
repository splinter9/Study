import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2] 
# 셔플 안하면 값이 개판된다

x_new = x[:-30]
y_new = y[:-30]
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18 <-서른개 줄었음
print(y_new)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=66, stratify=y_new)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('model_score : ', score)
y_predict = model.predict(x_test)
print("accuracy score : ", round(accuracy_score(y_test, y_predict), 4))

# 그냥 실행
# model_score :  0.9777777777777777
# accuracy score :  0.9778

# 서른개 줄였을때 / 데이터 축소
# model_score :  0.9459459459459459
# accuracy score :  0.9459


print('================ SMOTE 적용 ======================')

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('model_score : ', score)
y_predict = model.predict(x_test)
print("accuracy score : ", round(accuracy_score(y_test, y_predict), 4))

# 데이터 증폭
# model_score :  0.972972972972973
# accuracy score :  0.973

