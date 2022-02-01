import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 
print(pd.Series(y).value_counts())

print(y)


x_new = x[:-30]
y_new = y[:-30]
print(pd.Series(y_new).value_counts())

print(y_new)


x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=66, stratify=y_new)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('model_score : ', score)
y_predict = model.predict(x_test)
print("accuracy score : ", round(accuracy_score(y_test, y_predict), 4))




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




'''
model_score :  0.9629629629629629
accuracy score :  0.963

model_score :  0.9555555555555556
accuracy score :  0.9556
'''
