'''
PCA는 차원축소로 자원절약과 성능향상을 노림
칼럼을 압축하여 
CF) 임베딩으로 벡터화 하는것
 '''

from unittest import result
from inflection import dasherize
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
import sklearn as sk

#1. DATA
datasets = load_breast_cancer()
#datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
(506, 8)

pca = PCA(n_components=8)  #30개에서 8개로 줄였다
x = pca.fit_transform(x)
print(x.shape) #(569, 30) -> (569, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=66)


#2.MODEL
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3.FIT
model.fit(x_train, y_train, eval_metric='error')

#4.COMPILE
result = model.score(x_test, y_test)
print('결과:', result)


'''
(506, 13)
(506, 8)
결과: 0.7856968255504542
'''

# print(sk.__version__) #1.0.1


