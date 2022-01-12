import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
import seaborn as sns

#1. 데이터
path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA.csv")


data = dataset.drop(['STD_YM', 'CLSD_RATIO'], axis=1).values
target = dataset['CLSD_RATIO'].values
dataset.astype('float')
x = dataset.drop(['STD_YM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO','MED_ARR_AMT','CLSD_CNT','CLSD_RATIO'],axis=1)
y = dataset['CLSD_RATIO']

y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y = pd.get_dummies(y)

dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test, label=y_test)

params = {'max_depth' : 3,
          'eta': 0.1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          }
num_rounds = 400

#피처임포턴스
wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params=params, 
                      dtrain=dtrain, num_boost_round=num_rounds,
                      early_stopping_rounds=100, evals=wlist)

from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
plt.show()


#2.모델

model1 = RandomForestRegressor(n_estimators = 100, max_depth=10, min_samples_split=40, min_samples_leaf =30)
model2 = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30)
model3 = ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=7)
model4 = AdaBoostRegressor(n_estimators=100, random_state=7)
model5 = XGBRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30)
model6 = LGBMRegressor(n_estimators = 100, learning_rate = 0.1, max_depth=10, min_samples_split=40, min_samples_leaf =30)
model7 = CatBoostRegressor(n_estimators=100, max_depth=16, random_state=7)

from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(estimators=[('RandomForestRegressor', model1),
                                            ('GradientBoostingRegressor', model2),
                                            ('ExtraTreesRegressor', model3),
                                            ('AdaBoostRegressor', model4),
                                            ('XGBRegressor', model5),
                                            ('LGBMRegressor', model6),
                                            ('CatBoostRegressor', model7)],) #voting='hard')

classifiers = [model1,model2,model3,model4,model5,model6,model7]
from sklearn.metrics import r2_score

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    #loss = classifier.evaluate(x_test, y_test)
    r2 = r2_score(y_test, y_predict)
          
    class_name = classifier.__class__.__name__
    print("========== " + class_name + " ================")
    #print('loss : ', round(loss,3))
    print('r2 스코어 : ', round(r2,3))
    print('예측값 : ', y_predict[-1])


'''    
============== RandomForestRegressor ==================
r2 스코어 :  -0.15
예측값 :  0.48577370996233626

============== GradientBoostingRegressor ==================
r2 스코어 :  -0.288
예측값 :  0.44354134808433365

============== ExtraTreesRegressor ==================
r2 스코어 :  -0.094
예측값 :  0.43039407613095476

============== AdaBoostRegressor ==================
r2 스코어 :  -0.135
예측값 :  0.49255610667641825

============== XGBRegressor ==================
r2 스코어 :  -0.491
예측값 :  0.409935

============== LGBMRegressor ==================
r2 스코어 :  -0.077
예측값 :  0.4466462176614477
Learning rate set to 0.192834
0:      learn: 0.1944606        total: 686ms    remaining: 1m 7s
1:      learn: 0.1862210        total: 695ms    remaining: 34.1s
2:      learn: 0.1780452        total: 846ms    remaining: 27.4s
...
97:     learn: 0.0095195        total: 1m 13s   remaining: 1.49s
98:     learn: 0.0090667        total: 1m 14s   remaining: 749ms
99:     learn: 0.0086288        total: 1m 15s   remaining: 0us

============== CatBoostRegressor ==================
r2 스코어 :  -0.101
예측값 :  0.4560642226046058
'''
