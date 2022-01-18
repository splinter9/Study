from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score




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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2.모델
# model = Perceptron()
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = LogisticRegression()
# model = LinearRegression()
# model = DecisionTreeClassifier()
model = DecisionTreeRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# print("Perceptron : ", result)
# print("LinearSVC : ", result)
# print("SVC : ", result)
# print("KNeighborsClassifier : ", result)
# print("KNeighborsRegressor : ", result)
# print("LogisticRegression : ", result)
# print("LinearRegression : ", result)
# print("DecisionTreeClassifier : ", result)
print("DecisionTreeRegressor : ", result)
# print("RandomForestClassifier : ", result)
# print("RandomForestRegressor : ", result)

'''
r2스코어 :  -0.20852457863094087
DecisionTreeRegressor :  -0.20852457863094087
'''