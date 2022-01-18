from sklearn.linear_model import Perceptron
from sklearn.utils import all_estimators
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #분류모델에서 왠 회귀모델 이름?? 그러나 분류모델이다
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

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

# from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler
# scalar.fit(x_train)
# x_train = scalar.transform(x_train)
# x_test = scalar.transform(x_test)


#2.MODEL
# allAlgorithms = all_estimators(type_filter='classifier')    # 모델의 갯수 :  41
allAlgorithms = all_estimators(type_filter='regressor')   # 모델의 갯수 :  55

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수:", len(allAlgorithms)) #41

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name,'의 정답률 : ', r2)
    except:
        # continue
        print(name,'은 에러 터진 놈!!!!')
