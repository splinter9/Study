from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


datasets = load_wine()

# x = datasets.data
# y = datasets.target

x = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(x)


kmeans = KMeans(n_clusters=3, random_state=66) # 클러스터 숫자는 군집 구간 갯수
kmeans.fit(x)

print(kmeans.labels_)

print(datasets.target)

x['cluster'] = kmeans.labels_
x['target'] = datasets.target

print(accuracy_score(datasets.target, kmeans.labels_))


