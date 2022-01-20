#subplot을 이용하여 4개의 모델을 한 화면에 그래프로 그리기

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1.데이터

datasets = load_breast_cancer()
#print(datasets.DESCR)
#print(datasets.feature_names) 
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#피처임포턴스 확인결과 첫째 피처가 중요도0으로 나와서 빼주겠다!!

# x = pd.DataFrame(x) 판다스 drop을 사용해도 되고
# x = x.drop([0],axis=1)

x = datasets.data
# x = np.delete(x,[0,1],axis=1) #넘파이 delete를 써도 된다
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier



model1 = DecisionTreeClassifier(max_depth=5) 
model2 = RandomForestClassifier(max_depth=5)  #각각 하나씩 다 실습해보기
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()


#3. 컴파일, 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측
# result = model1.score(x_test, y_test) #아이리스는 분류모델이기에 모델스코어는 ACC


# from sklearn.metrics import accuracy_score
# y_predict = model1.predict(x_test)
# acc = accuracy_score(y_test, y_predict)



# print(result) #모델이 알아서 분류라서 acc 값으로 나온다
# print("acc:", acc)

# ###############피처 4개를 각각 빼본 결과###################
# print(model1.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plt.subplot(2,2,1) #2행2열의 첫번째 그래프
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2) #2행2열의 두번째 그래프
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3) #2행2열의 세번째 그래프
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4) #2행2열의 네번째 그래프
plot_feature_importances_dataset(model4)


# plot_feature_importances_dataset(model)
plt.show()
############################################################


################### 모델별 그래프 그리기 ####################
model_list = [model1,model2,model3,model4]
model_name = ['DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','XGBClassifier']
for i in range(4):
    plt.subplot(2, 2, i+1)
    model_list[i].fit(x_train, y_train)

    result = model_list[i].score(x_test, y_test)
    feature_importances_ = model_list[i].feature_importances_

    y_predict = model_list[i].predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("result", result)
    print("accuracy_score", acc)
    print("feature_importances_", feature_importances_)
    plot_feature_importances_dataset(model_list[i])
    plt.ylabel(model_name[i])

plt.show()
#####################################################
