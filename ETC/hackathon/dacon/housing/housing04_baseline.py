import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = '../_data/dacon/housing/' 
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')
train = train.iloc[:, 1:]
test = test.iloc[:, 1:]
# train.info()
# train.describe()
# test.describe()
train.drop_duplicates()
train = train[train['Garage Yr Blt'] < 2050]
# train.loc[254, 'Garage Yr Blt'] = 2007

cat_cols = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual']

for c in cat_cols :
    ord_df = train.groupby(c).target.median().reset_index(name = f'ord_{c}')
    train = pd.merge(train, ord_df, how = 'left')
    test = pd.merge(test, ord_df, how = 'left')

train.drop(cat_cols, axis = 1, inplace = True)
test.drop(cat_cols, axis = 1, inplace = True)

x = train.drop('target', axis = 1)
y = np.log1p(train.target)

target = test[x.columns]

target.fillna(target.mean(), inplace = True)

# # scaler = StandardScaler()
# # scaler = MinMaxScaler()
# # scaler = RobustScaler()
# # scaler = MaxAbsScaler()
# # scaler = QuantileTransformer()
# # scaler = PowerTransformer(method = 'box-cox') # error
# scaler = PowerTransformer(method = 'yeo-johnson') # 디폴트

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_sets = scaler.transform(test_sets)


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

def NMAE(true, pred) -> float:
    mae = np.mean(np.abs(true - pred))
    score = mae / np.mean(np.abs(true))
    return score

nmae_score = make_scorer(NMAE, greater_is_better=False)

kf =KFold(n_splits= 20, random_state= 77, shuffle=True)

'''
### XGBOOST
xgb_pred = np.zeros(target.shape[0])
xgb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    tr_data = Pool(data = tr_x, label = tr_y)
    val_data = Pool(data = val_x, label = val_y)
    
    xgb = XGBRegressor(depth = 4, random_state = 42, n_estimators = 3000, learning_rate = 0.03, verbose = 0)
    xgb.fit(tr_data, val_data) # early_stopping_rounds = 750, verbose = 1000)
    
    val_pred = np.expm1(xgb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    xgb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = xgb.predict(target) / 10
    xgb_val += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(xgb_val)} & std = {np.std(xgb_val)}')
'''

### RandomForest
rf_pred = np.zeros(target.shape[0])
rf_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rf = RandomForestRegressor(random_state = 77, criterion = 'mae')
    rf.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rf.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rf_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = rf.predict(target) / 20
    rf_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rf_val)} & std = {np.std(rf_val)}')



### GradientBoosting
gbr_pred = np.zeros(target.shape[0])
gbr_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    gbr = GradientBoostingRegressor(random_state = 77, max_depth = 5, learning_rate = 0.03, n_estimators = 3000)
    gbr.fit(tr_x, tr_y)
    
    val_pred = np.expm1(gbr.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    gbr_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = gbr.predict(target) / 20
    gbr_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(gbr_val)} & std = {np.std(gbr_val)}')


### CatBoost
cb_pred = np.zeros(target.shape[0])
cb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    tr_data = Pool(data = tr_x, label = tr_y)
    val_data = Pool(data = val_x, label = val_y)
    
    cb = CatBoostRegressor(depth = 5, random_state = 77, loss_function = 'MAE', n_estimators = 3000, learning_rate = 0.03, verbose = 0)
    cb.fit(tr_data, eval_set = val_data, early_stopping_rounds = 800, verbose = 0)
    
    val_pred = np.expm1(cb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    cb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = cb.predict(target) / 20
    cb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(cb_val)} & std = {np.std(cb_val)}')


### NGBoost
ngb_pred = np.zeros(target.shape[0])
ngb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(x, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = x.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    ngb = NGBRegressor(random_state = 77, n_estimators = 3000, verbose = 0, learning_rate = 0.03)
    ngb.fit(tr_x, tr_y, val_x, val_y, early_stopping_rounds = 800)
    
    val_pred = np.expm1(ngb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    ngb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = ngb.predict(target) / 20
    ngb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(ngb_val)} & std = {np.std(ngb_val)}') 






(ngb_pred + cb_pred + rf_pred + gbr_pred) / 4

submission['target'] = np.expm1((ngb_pred + cb_pred + rf_pred + gbr_pred) / 4)

submission.to_csv(path +'3rd.csv', index = False)