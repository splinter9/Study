from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer #
from sklearn.preprocessing import PowerTransformer    #
from sklearn.preprocessing import PolynomialFeatures  #

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8) #stratify=y<회기모델이라서 못쓴다>

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. MODEL
# model = XGBRegressor()
model = XGBRegressor(n_estimator=200,
                     learning_rate=0.01,
                     n_jobs=-1)


#3. FIT
import time
start = time.time()
model.fit(x_train, y_train, verbose=1)
end = time.time()
print('걸린시간:', end - start)

#4. COMPILE
results = model.score(x_test, y_test)


print(results)

