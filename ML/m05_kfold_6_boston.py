import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print(scores)
print("r2 : ", scores, "\n cross_val_score : ", np.mean(scores),4)


'''
r2 :  [0.88512503 0.71539805 0.80148666 0.87036533 0.89328401] 
cross_val_score :  0.833131817048082 4
'''
