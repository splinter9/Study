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


pca = PCA(n_components=10)  
x = pca.fit_transform(x)
print(x.shape) #(569, 30) -> (569, 8)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) #10개 요소의 총 합은 1이다. 즉, 중요도의 비율이다.
# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07]
print(sum(pca_EVR)) #0.9999998946838408

cumsum = np.cumsum(pca_EVR) 
print(cumsum) #누적합
#[0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453 0.99999854 0.99999936 0.99999971 0.99999989]

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.plot(pca_EVR)
plt.grid()
plt.show()

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


