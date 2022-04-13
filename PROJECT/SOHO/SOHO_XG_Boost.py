import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
 
path = '../_data/project/'
dft = dataset = pd.read_csv(path + "SOHO_DATA_T.csv")

# print(dft.info())

dft_X = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO'],axis=1)
dft_Y = dataset['CLSD_CNT']


dft_x_train, dft_x_test, dft_y_train, dft_y_test=train_test_split(dft_X, dft_Y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsampel=0.75, colsample_bytree=1, max_depth=7)

#print(len(dft_x_train), len(dft_x_test))
xgb_model.fit(dft_x_train, dft_y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
             missing=None, n_estimators=10000, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=True, subsample=0.75)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
plt.show()

pred=xgb_model.predict(dft_x_test)

print(pred)

r_sq=xgb_model.score(dft_x_train, dft_y_train)
print(r_sq)
print(explained_variance_score(pred, dft_y_test))


