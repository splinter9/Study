#########################################################
############### 지역별 SOHO 폐업률 예측 LSTM #############
#########################################################  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy


#1. 데이터
path = '../_data/project/'
dataset = pd.read_csv(path + "SOHO_DATA_h.csv")

dataset.describe()
print(dataset.columns)
print(dataset.info())

# Data columns (total 42 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   STD_YM            187 non-null    int64
#  1   BLCK_SP_CD        187 non-null    int64
#  2   CTPV_CD           187 non-null    int64
#  3   CTPV_NM           187 non-null    object
#  4   AVG_ICYR_AMT      187 non-null    int64
#  5   MED_ICYR_AMT      187 non-null    int64
#  6   POP_CT            187 non-null    int64
#  7   CARD_HLDR_CT      187 non-null    int64
#  8   AVG_CARD_CNT      187 non-null    int64
#  9   AVG_CARD_LMT_AMT  187 non-null    int64
#  10  CARD_USE_AMT      187 non-null    int64
#  11  CRD_SLE_USE_AMT   187 non-null    int64
#  12  LSP_USE_AMT       187 non-null    int64
#  13  INSTL_USE_AMT     187 non-null    int64
#  14  CASH_SVC_SUE_AMT  187 non-null    int64
#  15  OVSEA_CSMP_AMT    187 non-null    int64
#  16  LN_HLDR_CT        187 non-null    int64
#  17  AVG_LN_CONT       187 non-null    int64
#  18  ALL_LN_BLC        187 non-null    int64
#  19  AVG_LN_BLC        187 non-null    int64
#  20  ALL_ARR_HLDR_CT   187 non-null    int64
#  21  AVG_ARR_CONT      187 non-null    int64
#  22  AVG_ARR_DACT      187 non-null    int64
#  23  AVG_ARR_AMT       187 non-null    int64
#  24  MED_ARR_AMT       187 non-null    int64
#  25  DEPOSIT_AMT       187 non-null    int64
#  26  AVG_ICYR_ICM      187 non-null    int64
#  27  CARD_USE_AMT.1    187 non-null    int64
#  28  RE_DEBT_AMT       187 non-null    int64
#  29  TOT_ASST_AMT      187 non-null    int64
#  30  NET_ASST_AMT      187 non-null    int64
#  31  ONW_HOUS_RATIO    187 non-null    float64
#  32  PLU_HOUS_RATIO    187 non-null    float64
#  33  APT_RES_RATIO     187 non-null    float64
#  34  OPN_CNT           187 non-null    int64
#  35  CLSD_CNT          187 non-null    int64
#  36  CLSD_RATIO        187 non-null    float64
#  37  NEW_CNT           187 non-null    int64
#  38  MANAGE01_CNT      187 non-null    int64
#  39  MANAGE02_CNT      187 non-null    int64
#  40  MANAGE03_CNT      187 non-null    int64
#  41  MANAGE04_CNT      187 non-null    int64
# dtypes: float64(4), int64(37), object(1)


plt.figure(figsize=(20,10))
heat_table = dataset.drop(['STD_YM'], axis=1).corr()
heatmap_ax = sns.heatmap(heat_table, annot=True, cmap='coolwarm')
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=8, rotation=90)
heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=8)
plt.title('Correlation between Features', fontsize=30)
plt.show()


from sklearn.preprocessing import StandardScaler 

# 데이터 표준화
data = dataset.drop(['STD_YM', 'CLSD_RATIO','CTPV_NM'], axis=1).values
target = dataset['CLSD_RATIO'].values
scaled_data = StandardScaler().fit_transform(data)


# 2D 시각화하기 위해 주성분을 2개로 선택합니다.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
principalDf = pd.DataFrame(data=pca_data, columns = ['principal component 1', 'principal component 2'])
principalDf['target'] = target

plt.figure(figsize = (20, 15))

plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 component PCA', fontsize = 40)

sns.scatterplot(x='principal component 1', y='principal component 2', data=principalDf, hue='target', s= 100)
plt.show()


x = dataset.drop(['STD_YM','CTPV_NM','BLCK_SP_CD','CTPV_CD','ONW_HOUS_RATIO','PLU_HOUS_RATIO','APT_RES_RATIO','MED_ARR_AMT','CLSD_CNT','CLSD_RATIO'],axis=1)
y = dataset['CLSD_RATIO']


x, y = x.values, y.values
y = np.log1p(y)
print(dataset.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)


#피처임포턴스
dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test, label=y_test)

params = {'max_depth' : 3,
          'eta': 0.1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          }
num_rounds = 400

wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params=params, 
                      dtrain=dtrain, num_boost_round=num_rounds,
                      early_stopping_rounds=100, evals=wlist)

from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
plt.show()


#1-1 데이터표준화
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


#1-2 결측치확인
def check_missing_col(dataframe):
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'총 {missing_values}개의 결측치가 존재합니다.')

        if i == len(dataframe.columns) - 1 and counted_missing_col == 0:
            print('결측치가 존재하지 않습니다') 

check_missing_col(dataset)

#결측치가 존재하지 않습니다




#1-3 리쉐이프
print(x.shape, y.shape) #(187, 32) (187,)
x = x.reshape(187, 32, 1)




#2. 모델구성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(32,1))) 
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(160, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(140, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련

import time
start = time.time()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

end = time.time() - start

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict[-1])
 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)        # r2 보다 loss가 더 신뢰성높음??
print('r2 스코어 : ', round(r2,3))

print('걸린시간:', round(end,3), '초')


'''
loss :  642217.5625
예측값 :  [623.80035]
걸린시간: 896.292 초

loss: 234045.4375
예측값 :  [87293.06]
r2 스코어 :  -125693.659005371
걸린시간: 45.989 초

loss: 0.5437342524528503
예측값 :  [[7.0142355]
r2 스코어 :  -0.00023665574980791249
걸린시간: 6422.191 초

loss: 0.5437701344490051
예측값 :  [7.0127506]
r2 스코어 :  -0.0003026751135752903
걸린시간: 6243.682 초

loss: 1.857051968574524
예측값 :  [5.879521]
r2 스코어 :  -2.416
걸린시간: 1928.597 초

loss: 0.54374098777771
예측값 :  [7.013943]
r2 스코어 :  -0.0
걸린시간: 3330.395 초

loss: 0.5437672138214111
예측값 :  [7.0128646]
r2 스코어 :  -0.0
걸린시간: 3232.33 초

loss: 0.0064612398855388165
예측값 :  [0.4637093]
r2 스코어 :  -0.079
걸린시간: 2751.36 초

'''




'''
le = LabelEncoder()
le.fit(x)
#y = to_categorical(y)
print(x.shape)      #(187, 40)
print(y.shape)      #(187, 37286)

x_train, x_test, y_train, y_test = train_test_split(x,y,
         train_size =0.7, shuffle=True, random_state = 42)  

# #데이터표준화
# mean = np.mean(x_train, axis=0)
# std = np.sdt(x_train, axis=0)

# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std

print(x_train.shape, y_train.shape) #(130, 40) (130, 37286)
print(x_test.shape, y_test.shape) #(57, 40) (57, 37286)
#130개의 학습 데이터와 57개의 테스트




#검증데이터셋
 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# scaler = StandardScaler()         
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(200, input_dim=34)) 
#model.add(LSTM(32,activation='relu',input_shape = (3,34)))
model.add(Dense(130, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
                        filepath = './_ModelCheckPoint/keras27_1_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, callbacks=[es, mcp])



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
acc = model.evaluate(x_test, y_test)[1]
print("loss, acc : ", loss)


y_pred = model.predict(x_test)
print(y_pred[-1])
'''
