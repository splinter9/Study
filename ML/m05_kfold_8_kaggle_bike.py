import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # Classifier :분류모델
from sklearn.linear_model import LogisticRegression,LinearRegression        # LogisticRegression :분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score

#1. 데이터 
path = './_data/bike/'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1)  
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

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
r2 :  [0.27289353 0.28846805 0.27974322 0.30256055 0.25805036] 
 cross_val_score :  0.2803431428196711 4
'''
