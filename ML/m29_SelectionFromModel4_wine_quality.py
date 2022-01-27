# GridSearchCV 적용 출력한 값에 모델 적용
# 피처임포턴스 추출후 
# SelectFromModel 만들기
# 칼럼 축소 후
# 모델 구축해서 결과 비교

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
path = 'D:\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',
                       index_col=None,
                       header=0,
                       sep=';', 
                       dtype=float)
datasets = datasets.values
print(type(datasets))
print(datasets.shape)

x = datasets[:, :11]
y = datasets[:,  11]

print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},]
model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score:', score)
y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))

# model.score: 0.869392356479609
# accuracy_score: 0.869392356479609

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
aaa = np.sort(model.feature_importances_)


print("======================================")
for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    print(selection_x_train.shape, selection_x_test.shape)
    
    selction_model = XGBRegressor(n_jobs=-1)
    selction_model.fit(selection_x_train, y_train)
    
    y_predict = selction_model.predict(selection_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, selection_x_train.shape[1], score*100))
    
'''
======================================
(3918, 11) (980, 11)
Thresh=0.069, n=11, R2: 46.54%
(3918, 10) (980, 10)
Thresh=0.070, n=10, R2: 45.82%
(3918, 9) (980, 9)
Thresh=0.070, n=9, R2: 44.90%
(3918, 8) (980, 8)
Thresh=0.071, n=8, R2: 46.99%
(3918, 7) (980, 7)
Thresh=0.072, n=7, R2: 45.22%
(3918, 6) (980, 6)
Thresh=0.072, n=6, R2: 41.91%
(3918, 5) (980, 5)
Thresh=0.074, n=5, R2: 38.39%
(3918, 4) (980, 4)
Thresh=0.085, n=4, R2: 37.15%
(3918, 3) (980, 3)
Thresh=0.088, n=3, R2: 34.13%
(3918, 2) (980, 2)
Thresh=0.115, n=2, R2: 23.20%
(3918, 1) (980, 1)
Thresh=0.214, n=1, R2: 20.84%
'''    