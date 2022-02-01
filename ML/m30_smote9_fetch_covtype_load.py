#증폭한 후 저장된 데이터를 불러와서 완성할것

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

print(pd.Series(y).value_counts())

# (581012, 54) (581012,)
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

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
    
