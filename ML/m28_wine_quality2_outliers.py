import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score

#1. DATA
path = 'D:\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',
                       index_col=None,
                       header=0,
                       sep=';', 
                       dtype=float)

print(datasets.shape)
print(datasets.describe())
print(datasets.info())

datasets = datasets.values
print(type(datasets))
print(datasets.shape)

x = datasets[:, :11]
y = datasets[:,  11]

print('라벨:',np.unique(y, return_counts=True))

#############################################################
#################### 아웃라이어 확인 1 #######################
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25, 50, 75])
    print("1사분위 :", quartile_1)
    print("q2:", q2)
    print("3사분위:",  quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))  # | = or

outliers_loc = outliers(datasets)
print("이상치의 위치 :" , outliers_loc)


import matplotlib.pyplot as plt
plt.boxplot(datasets)
plt.show()

plt.boxplot(outliers_loc, sym="bo")
plt.title('outliers_loc')
plt.show()

#############################################################
#################### 아웃라이어 확인2 ########################
def boxplot_vis(data, target_name):
    plt.figure(figsize=(30, 30))
    for col_idx in range(len(data.columns)):
        # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
        plt.subplot(6, 2, col_idx+1)
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
        # 그래프 타이틀: feature name
        plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
    # plt.savefig('../figure/boxplot_' + target_name + '.png')
    plt.show()
boxplot_vis(datasets,'white_wine')

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier
#############################################################



x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs = -1,
                     n_estimators = 2000,
                     learning_rate = 0.025,
                     max_depth = 5,
                     min_child_weight = 10,
                     subsample = 1,
                     colsample_bytree = 1,
                     reg_alpha = 1,      
                     reg_lambda = 0,
                     tree_method='gpu_hist',
                     predictor='gpu_predictor',
                     gpu_id = 0)

model.fit(x_train, y_train, verbose=1)


model_scores = model.score(x_test, y_test)
print("model_scores" , round(model_scores, 4))

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score :", f1)

