import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터

x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)
# x = pd.DataFrame(x, columns=datasets['feature_names'])
# print(x.feature_names)
x = pd.DataFrame(x)
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x.columns = feature_names
print(x)
x = x.drop(['age', 'sex', 's1', 's4'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8,) # stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score:', score)

# y_predict = model.predict(x_test)
# print('r2_score:', r2_score(y_test, y_predict))


print(model.feature_importances_)
print(np.sort(model.feature_importances_))
aaa = np.sort(model.feature_importances_)
#[0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319 0.06597802 0.07382318 0.19681741 0.39979857]

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
# (353, 10) (89, 10)
# Thresh=0.026, n=10, R2: 23.96%
# (353, 9) (89, 9)
# Thresh=0.033, n=9, R2: 27.03%
# (353, 8) (89, 8)
# Thresh=0.038, n=8, R2: 23.87%
# (353, 7) (89, 7)
# Thresh=0.048, n=7, R2: 26.48%
# (353, 6) (89, 6)
# Thresh=0.055, n=6, R2: 30.09%   ====> 상위 6개 피처로 돌렸을 때 상관관계가 제일 높다
# (353, 5) (89, 5)
# Thresh=0.063, n=5, R2: 27.41%
# (353, 4) (89, 4)
# Thresh=0.066, n=4, R2: 29.84%
# (353, 3) (89, 3)
# Thresh=0.074, n=3, R2: 23.88%
# (353, 2) (89, 2)
# Thresh=0.197, n=2, R2: 14.30%
# (353, 1) (89, 1)
# Thresh=0.400, n=1, R2: 2.56%

# ?? 중요도 순서대로 빼본거지 최선이라 볼 수 없다
# 피처를 선택해서 뺄 수 있는 방법은?


# 0.5 이상으로 만들것
