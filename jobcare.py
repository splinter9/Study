from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')


path = "../_data/dacon/jobcare/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")


# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

train.head()
test.head()

train = train.drop(['id', 'contents_open_dt'], axis=1) 
test = test.drop(['id', 'contents_open_dt'], axis=1)

model = RandomForestClassifier(n_estimators=400, max_depth=100, n_jobs=-1, verbose=1)

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)

preds = model.predict(test)

submit_file['target'] = preds


sample_submission = pd.read_csv(f'{path}sample_submission.csv')
sample_submission['target'] = preds
print(sample_submission)
sample_submission.to_csv('subm.csv', index=False)