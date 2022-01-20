from inspect import Parameter
from tkinter import Scale
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier


#1.데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names) 


x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingGridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
parameters = [
    {'randomforestclassifier__max_depth' : [6,8,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7],
    'randomforestclassifier__min_samples_split' : [3,5,10]}
]

parameters = [
    {'rf__max_depth' : [6,8,10]},
    {'rf__min_samples_leaf' : [3,5,7],
    'rf__min_samples_split' : [3,5,10]}
]


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline

#model = SVC() 
pipe = Pipeline(MinMaxScaler(), RandomForestClassifier())
#model = Pipeline([("mm", MinMaxScaler()),("svc", SVC())])
#model = GridSearchCV(pipe, parameters, cv=5, verbose=1)

#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()


#4. 평가, 예측

result = model.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('걸린시간 : ', end - start)
print("model.score : ", result) #모델이 알아서 분류라서 acc 값으로 나온다
print('acc:', acc)

