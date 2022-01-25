import numpy as np
import pandas as pd
from sklearn import impute

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

print(data.shape) #(4,5)
data = data.transpose()
data.columns = ['a', 'b', 'c', 'd']
print(data)
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   NaN  4.0   4.0  4.0
# 2   NaN  NaN   NaN  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# imputer = SimpleImputer(strategy='mean') #평균
# imputer = SimpleImputer(strategy='median') #중위
# imputer = SimpleImputer(strategy='most_frequnent') #최빈
# imputer = SimpleImputer(strategy='constant') #
imputer = SimpleImputer(strategy='constant', fill_value=777)


imputer.fit(data)
data2 = imputer.transform(data)
print(data2)

# fit에는 DataFrame이 들어가는데, 칼럼만 바꾸고 싶다면??
# 시리즈를 넣으면 에러가 난다
# 문제를 해결해 보시오

means = data['a'].mean()
