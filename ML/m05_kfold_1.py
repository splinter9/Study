from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

dataset = load_iris()
x = dataset.data
y = dataset.target

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)
print("ACC:", scores, "\n cross_val_score :", round(np.mean(scores),4))