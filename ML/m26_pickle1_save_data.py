from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_covtype, load_boston
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
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.9) #stratify=y<회기모델이라서 못쓴다>

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#데이터를 저장해보자
import pickle
path = './_save/'
pickle.dump(datasets, open(path + 'm23_pickle1_save_datasets.dat','wb'))



#2. MODEL
# model = XGBRegressor()
model = XGBClassifier(n_estimators = 2000,
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
          eval_metric='mlogloss', # rmse, mae, logloss, error, merror, mlogloss, uac, ndcg, map
          early_stopping_rounds=200, #기준은 eval_metric
          )
end = time.time()
print('걸린시간:', end - start)

#4. COMPILE/EVALUATE
results = model.score(x_test, y_test)
print("results:", round(results, 4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc:", acc)


print("=======================")
hist = model.evals_result()
print(hist)


# #저장
# import pickle
# path = './_save/'
# pickle.dump(model, open(path + 'm23_pickle1_save.dat','wb'))


















