'''
LDA
 '''

from unittest import result
from inflection import dasherize
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action="ignore")
import sklearn as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. DATA
#datasets = load_breast_cancer()
#datasets = fetch_california_housing()
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print(x.shape) #(581012, 54)

# pca = PCA(n_components=8)  #30개에서 8개로 줄였다
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis()
x = lda.fit_transform(x,y) #lda는 y값을 생성해주기 때문에 y도 핏해줘야함

print(x.shape) #(581012, 6)

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


