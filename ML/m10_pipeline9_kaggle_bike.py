from tkinter import Scale
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC


#1.데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names) 
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.pipeline import make_pipeline, Pipeline

print(x_train.shape, y_train.shape) #(120, 4) (120, 3)
print(x_test.shape, y_test.shape) #(30, 4) (30, 3)


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline

#model = SVC() 
model = make_pipeline(MinMaxScaler(),SVC())

#3. 컴파일, 훈련

model.fit(x_train, y_train)



#4. 평가, 예측

result = model.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)


print("model.score : ", result) #모델이 알아서 분류라서 acc 값으로 나온다
print('acc:', acc)


# 0.9666666666666667
# 0.9666666666666667
