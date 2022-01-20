import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler


#1.데이터

datasets = load_diabetes()
#print(datasets.DESCR)
#print(datasets.feature_names) 

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.pipeline import make_pipeline, Pipeline


#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline


# model = Perceptron() 
# model = LinearSVC() 
# model = SVC() 
# model = KNeighborsClassifier() 
# model = LogisticRegression() 
# model = DecisionTreeClassifier() 
# model = RandomForestClassifier()  #각각 하나씩 다 실습해보기

model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3. 컴파일, 훈련

model.fit(x_train, y_train)



#4. 평가, 예측

result = model.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)



print("model.score : ", result) #모델이 알아서 분류라서 acc 값으로 나온다
print('r2:', r2)

'''
model.score :  0.35668932865970615
r2: 0.35668932865970615
'''
