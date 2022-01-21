import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings(action="ignore")


#1. DATA
datasets = load_iris()
# datasets = load_breast_cancer()
# datasets = load_wine()
# datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape) 


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=66,
    stratify=y) #

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# pca = PCA(n_components=8)  #30개에서 8개로 줄였다
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
x = lda.fit_transform(x,y) #lda는 y값을 생성해주기 때문에 y도 핏해줘야함
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
print(x.shape)


#2.MODEL
from xgboost import XGBRegressor, XGBClassifier
model = XGBRegressor()
# model = XGBClassifier()

#3.FIT
model.fit(x_train, y_train, eval_metric='error')

#4.COMPILE
result = model.score(x_test, y_test)
print('결과:', result)


'''
XGBoost 기본
(569, 30)
(569, 30)
결과: 0.9736842105263158

PCA
(506, 13)
(506, 8)
결과: 0.7856968255504542

LDA
(569, 30)
(569, 1)
결과: 0.9824561403508771
'''


# print(sk.__version__) #1.0.1


