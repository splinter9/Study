### kaggle Titanic 문제 ###

import numpy as np
import pandas as pd
from sklearn import datasets

#1. 데이터
path = "./_data/titanic/"

train = pd.read_csv(path + "train.csv", index_col=0, header=0) #첫째열이 인덱스일 뿐이라 데어터로 못씀
print(train) 
print(train.shape) #(891, 11) =>첫째열을 인덱스로 바꿔서 12가 아니라 11

test = pd.read_csv(path + "test.csv")
gen = pd.read_csv(path +"gender_submission.csv", index_col=0, header=0)

print(test)
print(test.shape) #(418, 11)
print(gen.shape) #(418, 1)  =>첫째열을 인덱스로 바꿔서 2가 아니라 1
print(gen)





