import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1.데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names)

#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
print(x)
print(y)
print(type(x)) #<class 'numpy.ndarray'>

df = pd.DataFrame(x, columns = datasets['feature_names'])
df = pd.DataFrame(x, columns = datasets.feature_names)
df = pd.DataFrame(x, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
print(df)

df['Target'] = y
df['Target(Y)'] = y # Y칼럼 추가

print('===============상관계수 히트 맵===============')
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
#히트맵에서 숫자는 무슨의미? 
plt.show()

