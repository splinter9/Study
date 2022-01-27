# GridSearchCV 적용 출력한 값에 모델 적용
# 피처임포턴스 추출후 
# SelectFromModel 만들기
# 칼럼 축소 후
# 모델 구축해서 결과 비교


import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
x, y = fetch_covtype(return_X_y=True)
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8,) # stratify=y)

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
    

###### 결과 ######
# (464809, 54) (116203, 54)
# Thresh=0.001, n=54, R2: 74.35%
# (464809, 53) (116203, 53)
# Thresh=0.002, n=53, R2: 74.73%
# (464809, 52) (116203, 52)
# Thresh=0.003, n=52, R2: 74.73%
# (464809, 51) (116203, 51)
# Thresh=0.003, n=51, R2: 74.75%
# (464809, 50) (116203, 50)
# Thresh=0.003, n=50, R2: 75.30% ===> 
# (464809, 49) (116203, 49)
# Thresh=0.004, n=49, R2: 74.48%
# (464809, 48) (116203, 48)
# Thresh=0.005, n=48, R2: 74.34%
# (464809, 47) (116203, 47)
# Thresh=0.005, n=47, R2: 74.52%
# (464809, 46) (116203, 46)
# Thresh=0.005, n=46, R2: 74.07%
# (464809, 45) (116203, 45)
# Thresh=0.005, n=45, R2: 74.25%
# (464809, 44) (116203, 44)
# Thresh=0.006, n=44, R2: 74.14%
# (464809, 43) (116203, 43)
# Thresh=0.006, n=43, R2: 74.49%
# (464809, 42) (116203, 42)
# Thresh=0.006, n=42, R2: 74.68%
# (464809, 41) (116203, 41)
# Thresh=0.007, n=41, R2: 74.52%
# (464809, 40) (116203, 40)
# Thresh=0.007, n=40, R2: 74.37%
# (464809, 39) (116203, 39)
# Thresh=0.007, n=39, R2: 74.04%
# (464809, 38) (116203, 38)
# Thresh=0.008, n=38, R2: 74.52%
# (464809, 37) (116203, 37)
# Thresh=0.008, n=37, R2: 74.34%
# (464809, 36) (116203, 36)
# Thresh=0.009, n=36, R2: 74.95%
# (464809, 35) (116203, 35)
# Thresh=0.009, n=35, R2: 74.66%
# (464809, 34) (116203, 34)
# Thresh=0.010, n=34, R2: 73.95%
# (464809, 33) (116203, 33)
# Thresh=0.010, n=33, R2: 74.17%
# (464809, 32) (116203, 32)
# Thresh=0.010, n=32, R2: 73.87%
# (464809, 31) (116203, 31)
# Thresh=0.012, n=31, R2: 73.42%
# (464809, 30) (116203, 30)
# Thresh=0.012, n=30, R2: 73.12%
# (464809, 29) (116203, 29)
# Thresh=0.012, n=29, R2: 72.94%
# (464809, 28) (116203, 28)
# Thresh=0.012, n=28, R2: 66.56%
# (464809, 27) (116203, 27)
# Thresh=0.013, n=27, R2: 64.79%
# (464809, 26) (116203, 26)
# Thresh=0.013, n=26, R2: 63.65%
# (464809, 25) (116203, 25)
# Thresh=0.013, n=25, R2: 63.72%
# (464809, 24) (116203, 24)
# Thresh=0.014, n=24, R2: 53.43%
# (464809, 23) (116203, 23)
# Thresh=0.014, n=23, R2: 46.77%
# (464809, 22) (116203, 22)
# Thresh=0.015, n=22, R2: 46.58%
# (464809, 21) (116203, 21)
# Thresh=0.016, n=21, R2: 46.50%
# (464809, 20) (116203, 20)
# Thresh=0.017, n=20, R2: 45.78%
# (464809, 19) (116203, 19)
# Thresh=0.018, n=19, R2: 45.39%
# (464809, 18) (116203, 18)
# Thresh=0.019, n=18, R2: 45.17%
# (464809, 17) (116203, 17)
# Thresh=0.020, n=17, R2: 44.53%
# (464809, 16) (116203, 16)
# Thresh=0.022, n=16, R2: 44.53%
# (464809, 15) (116203, 15)
# Thresh=0.024, n=15, R2: 44.55%
# (464809, 14) (116203, 14)
# Thresh=0.025, n=14, R2: 44.32%
# (464809, 13) (116203, 13)
# Thresh=0.025, n=13, R2: 43.92%
# (464809, 12) (116203, 12)
# Thresh=0.027, n=12, R2: 43.90%
# (464809, 11) (116203, 11)
# Thresh=0.029, n=11, R2: 43.73%
# (464809, 10) (116203, 10)
# Thresh=0.034, n=10, R2: 43.40%
# (464809, 9) (116203, 9)
# Thresh=0.037, n=9, R2: 40.92%
# (464809, 8) (116203, 8)
# Thresh=0.039, n=8, R2: 39.43%
# (464809, 7) (116203, 7)
# Thresh=0.042, n=7, R2: 39.13%
# (464809, 6) (116203, 6)
# Thresh=0.045, n=6, R2: 36.28%
# (464809, 5) (116203, 5)
# Thresh=0.048, n=5, R2: 36.02%
# (464809, 4) (116203, 4)
# Thresh=0.049, n=4, R2: 35.90%
# (464809, 3) (116203, 3)
# Thresh=0.049, n=3, R2: 33.21%
# (464809, 2) (116203, 2)
# Thresh=0.059, n=2, R2: 32.55%
# (464809, 1) (116203, 1)
# Thresh=0.088, n=1, R2: 29.17%

