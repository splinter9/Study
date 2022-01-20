import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1.데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names) 
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#피처임포턴스 확인결과 첫째 피처가 중요도0으로 나와서 빼주겠다!!

# x = pd.DataFrame(x) 판다스 drop을 사용해도 되고
# x = x.drop([0],axis=1)

x = datasets.data
#x = np.delete(x,[0,1],axis=1) #넘파이 delete를 써도 된다
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# model = Perceptron() 
# model = LinearSVC() 
# model = SVC() 
# model = KNeighborsClassifier() 
# model = LogisticRegression() 
# model = DecisionTreeClassifier(max_depth=3) 
# model = RandomForestClassifier()  #각각 하나씩 다 실습해보기
model = XGBClassifier()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)



print(result) #모델이 알아서 분류라서 acc 값으로 나온다
print("acc:", acc)

print(model.feature_importances_)
