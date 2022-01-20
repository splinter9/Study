#subplot을 이용하여 4개의 모델을 한 화면에 그래프로 그리기
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




#1. 데이터 
path = 'D:\_data\kaggle\bike'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path + 'train.csv')
#print(train.shape)  # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
#print(test_file.shape)  # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
#print(submit_file.shape)  # (6493, 2)

x = train.drop(['datetime', 'casual','registered','count'], axis=1).values
#print(x.shape)  # (10886, 8)

y = train['count']
#print(y.shape)  # (10886,)

test_file = test_file.drop(['datetime'], axis=1)  


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 


# 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


model1 = DecisionTreeRegressor(max_depth=5) 
model2 = RandomForestRegressor(max_depth=5)  #각각 하나씩 다 실습해보기
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()


#3. 컴파일, 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


#4. 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 


# from sklearn.metrics import accuracy_score, r2_score
# y_predict = model1.predict(x_test)
# r2 = r2_score(y_test, y_predict)



# print(result) #모델이 알아서 분류라서 acc 값으로 나온다
# print("r2:", r2)

# ###############피처 4개를 각각 빼본 결과#####################
# print(model1.feature_importances_)

import matplotlib.pylab as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = train.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), train.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plt.subplot(2,2,1)  #2행2열의 첫번째 그래프
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2)  #2행2열의 두번째 그래프
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3)  #2행2열의 세번째 그래프
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4)  #2행2열의 네번째 그래프
plot_feature_importances_dataset(model4)


# plot_feature_importances_dataset(model)
plt.show()
#############################################################


# ################### 모델별 그래프 그리기 ####################
# model_list = [model1,model2,model3,model4]
# model_name = ['DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','XGBClassifier']
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     model_list[i].fit(x_train, y_train)

#     result = model_list[i].score(x_test, y_test)
#     feature_importances_ = model_list[i].feature_importances_

#     y_predict = model_list[i].predict(x_test)
#     acc = accuracy_score(y_test, y_predict)
#     print("result", result)
#     print("accuracy_score", acc)
#     print("feature_importances_", feature_importances_)
#     plot_feature_importances_dataset(model_list[i])
#     plt.ylabel(model_name[i])

# plt.show()
# #####################################################

