import numpy as np
import pandas as pd

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

# 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제 (pandas)
# print(data.dropna())
# print(data.dropna(axis=0)) #0은 행
# print(data.dropna(axis=1)) #1은 열

#2-1. 특정값 - 평균값
means = data.mean() #data평균
print(means)
data1 = data.fillna(means) #결측값을 평균값으로 채움
print(data)

#2-2. 특정값 - 중위값
meds = data.median()
print(meds)
data2 = data.fillna(meds)
print(data2)

#2-3. 특정값 - 프론트필 ffill, 백필 bfill
data2 = data.fillna(method='ffill')
print(data2) #이전 데이터 값을 채워준다
data2 - data.fillna(method='bfill')
print(data2) 

data2 = data.fillna(method='ffill', limit=1)
print(data2) 
data2 - data.fillna(method='bfill', limit=1)
print(data2)


#2-3. 특정값 - 채우기
data2 = data.fillna(828282)
print(data2)

#############  특정 칼럼별 결측치 처리  #############

means = data['a'].mean()
print(means)
data['a'] = data['a'].fillna(means)
print(data)

meds = data['b'].median()
print(meds)
data['b'] = data['b'].fillna(meds)
print(data)

meds = data['c'].median()
print(meds)
data['c'] = data['c'].fillna(meds)
print(data)
