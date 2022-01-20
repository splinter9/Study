import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
#y = np.append(y_train,y_test, axis=0)

print(x.shape)
x = x.reshape(70000,-1)
#x = x.reshape(x.shape[0], )
#############################################
#실습
#PCA를 통해 0.95 이상인 n_components 가 몇개??
# 28*28 이미지를 1차원인 784로 줄세운다
#0.95
#0.99
#0.999
#1.0
#np.argmax 사용할것
#############################################

pca = PCA(n_components=784)
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

