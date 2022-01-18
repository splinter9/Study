import numpy as np
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()


scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC : ", scores, "\n cross_val_score : ", np.mean(scores),4)

'''
ACC :  [0.96703297 0.96703297 0.97802198 0.92307692 0.92307692] 
 cross_val_score :  0.9516483516483516 4
'''
