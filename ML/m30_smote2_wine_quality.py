import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = 'D:\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',
                       index_col=None,
                       header=0,
                       sep=';', 
                       dtype=float)
datasets = datasets.values
print(type(datasets))   # <class 'numpy.ndarray'>
print(datasets.shape)   # (4898, 12)

x = datasets[:, :11]
y = datasets[:,  11]

print(x.shape, y.shape)            # (4898, 11) (4898,)
print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
print(y) # [6. 6. 6. ... 6. 7. 6.]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print('model_score : ', round(model.score(x_test, y_test),4))
print("accuracy score : ", round(accuracy_score(y_test, y_predict), 4))
print("f1_score :", round(f1_score(y_test, y_predict, average='macro'),4))



print('======================= SMOTE 적용 ============================')

smote = SMOTE(random_state=66, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)


print('model_score : ', round(model.score(x_test, y_test),4))
print("accuracy score : ", round(accuracy_score(y_test, y_predict), 4))
print("f1_score :", round(f1_score(y_test, y_predict, average='macro'),4))

# 그냥 했을때
# model_score :  0.643265306122449
# accuracy score :  0.6433

# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6

# 디폴트
# model_score :  0.6424
# accuracy score :  0.6424
# f1_score : 0.385

# 증폭
# model_score :  0.6367
# accuracy score :  0.6367
# f1_score : 0.3919
