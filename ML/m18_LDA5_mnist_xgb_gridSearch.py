import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.datasets import load_boston, fetch_california_housing, load_diabetes

from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier, XGBRegressor

import warnings
warnings.filterwarnings(action="ignore")


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
y = np.append(y_train,y_test, axis=0)

print(x.shape)
print(y.shape)

x = x.reshape(70000,-1)

pca = PCA(n_components=154)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

# 0이 포함되면 안되니까 1을 더해줌
print(np.argmax(cumsum >=0.95)+1)  # 154
print(np.argmax(cumsum >=0.99)+1)  # 331
print(np.argmax(cumsum >=0.999)+1) # 486
print(np.argmax(cumsum)+1)         # 713

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=66,
    stratify=y) #

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# pca = PCA(n_components=8)  #30개에서 8개로 줄였다
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
x = lda.fit_transform(x,y) #lda는 y값을 생성해주기 때문에 y도 핏해줘야함
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
print(x.shape)

#2.MODEL

model = GridSearchCV(XGBClassifier(), verbose=1, refit=True, n_jobs=-1) 
#Fitting 5 folds for each of 42 candidates, totalling 210 fits


#3.FIT
import time
start = time.time()
model.fit(x_train, y_train, eval_metric='error')


#4. COMPILE

result = model.score(x_test, y_test)
print('결과:', result)

end = time.time() - start
print('걸린시간:', round(end,3), '초')

