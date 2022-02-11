import tensorflow as tf
from sklearn.datasets import load_
from sklearn.metrics import accuracy_score
tf.set_random_seed(104)

x = Bikedata.drop(['casual','registered','count'], axis=1)  
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)
y = Bikedata['count']