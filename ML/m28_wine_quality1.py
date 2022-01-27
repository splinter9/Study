import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer 
from sklearn.preprocessing import PowerTransformer    
from sklearn.preprocessing import PolynomialFeatures  

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
path = 'D:\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv', delimiter=';', dtype=float)


x = datasets.drop(['quality'], axis=1) # 컬럼을 삭제할때는 axis=1, 디폴트값은 axis=0
y = datasets['quality']


print(x.shape, y.shape) #(4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25, 50, 75])
    print("1사분위 :", quartile_1)
    print("q2:", q2)
    print("3사분위:",  quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))  # | = or

outliers_loc = outliers(datasets)
print("이상치의 위치 :" , outliers_loc)


import matplotlib.pyplot as plt
plt.boxplot(datasets)
plt.show()

plt.boxplot(outliers_loc, sym="bo")
plt.title('outliers_loc')
plt.show()


#2. MODEL
# model = XGBRegressor()
model = XGBClassifier(n_jobs = -1,
                     n_estimators = 20000,
                     learning_rate = 0.025,
                     max_depth = 10,
                     min_child_weight = 10,
                     subsample = 1,
                     colsample_bytree = 1,
                     reg_alpha = 1,      
                     reg_lambda = 0,
                     tree_method='gpu_hist',
                     predictor='gpu_predictor',
                     gpu_id = 0)



#3. FIT
import time
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test), (x_test, y_test)],
          eval_metric='merror', # rmse, mae, logloss, error, merror, mlogloss, uac, ndcg, map
          early_stopping_rounds=300, #기준은 eval_metric
          ) 
end = time.time()
print('걸린시간:', end - start)

#4. COMPILE
results = model.score(x_test, y_test)
print("results" , round(results, 4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

# print("=======================")
# hist = model.evals_result()
# print(hist)

'''
걸린시간: 44.11556339263916
results 0.6643
acc : 0.6642857142857143


'''