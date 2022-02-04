from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()

# x = datasets.data
# y = datasets.target

irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(irisDF)


kmeans = KMeans(n_clusters=3, random_state=66) # 클러스터 숫자는 군집 구간 갯수
# 데이터의 N개 평균값을 구한후 N개의 집단으로 뭉쳐서 모으는것 

kmeans.fit(irisDF)

print(kmeans.labels_)
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2
#  2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 0 2 2 2 0 2 2 2 0 2
#  2 0]
print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# kmeans.labels_ = datasets.target 거의 유사하다

irisDF['cluster'] = kmeans.labels_
irisDF['target'] = datasets.target

print(accuracy_score(datasets.target, kmeans.labels_))


# iris_results = kmeans.score(irisDF)
# print("results" , round(iris_results, 4))

# iris_pred = kmeans.predict(irisDF)
# acc = accuracy_score(irisDF, iris_pred)
# print("acc :", acc)

# print("=======================")
# hist = kmeans.evals_result()
# print(hist)