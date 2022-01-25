from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
# import warnings
# warnings.filterwarnings(action='ignore')



#1. DATA
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.9) #stratify=y<회기모델이라서 못쓴다>

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. MODEL
# model = XGBRegressor()
model = XGBRegressor(n_estimators = 2000,
                     learning_rate = 0.025,
                     n_jobs = -1,
                     max_depth = 3,
                     min_child_weight = 10,
                     subsample = 1,
                     colsample_bytree = 1,
                     reg_alpha = 1,      #규제 L1
                     reg_lambda = 0,)    #규제 L2


#3. FIT
import time
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test), (x_test, y_test)],
          eval_metric='rmse', # rmse, mae, logloss, error, merror, mlogloss, uac, ndcg, map
          early_stopping_rounds=200, #기준은 eval_metric
          ) 
end = time.time()
print('걸린시간:', end - start)

#4. COMPILE/EVALUATE
results = model.score(x_test, y_test)
print("results:", round(results, 4))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2:", r2)


print("=======================")
hist = model.evals_result()
print(hist)


#저장
# import pickle
path = './_save/'
# pickle.dump(model, open(path + 'm23_pickle1_save','wb'))

import joblib
joblib.dump(model, path + 'm24_joblib1_save.dat')



'''
걸린시간: 2.1582562923431396
results: 0.9574
r2: 0.9574352488905513
'''




















'''
# 훈련 및 검증 손실 그리기
results = model.evals_result()
import matplotlib.pyplot as plt

train_error = results['validation_0']['rmse']
test_error = results['validation_1']['rmse']

epoch = range(1, len(train_error)+1)
plt.plot(epoch, train_error, label = 'Train')
plt.plot(epoch, test_error, label = 'Test')
plt.ylabel('Classification Error')
plt.xlabel('Model Complexity (n_estimators)')
plt.legend()
plt.show()
'''



'''
튠
(20640, 8) (20640,)
걸린시간: 91.36105728149414
results 0.8538

디폴트
(20640, 8) (20640,)
걸린시간: 1.0327916145324707
results 0.8434
'''
