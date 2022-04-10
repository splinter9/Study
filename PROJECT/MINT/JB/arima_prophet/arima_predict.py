import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2020-02-29') #60 2-29 / 20 1-21 / 120 4-30
dates = pd.date_range(start_date,end_date,freq='D')[::-1]

f_1m = pd.read_csv("CF 202206_1M.csv")
f1m = pd.DataFrame()
# print(f_1m)
f1m["Close"] = f_1m["종가"]
f1m2 = f1m.loc[0:59]
# print(f1m2.head())
f1m2.index = dates
f1m2.index.name = "day"
ts_log = f1m2[::-1]
# train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
train_data = ts_log #, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]


# plt.figure(figsize=(13,6))
# plt.grid(True)
# plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
# plt.plot(test_data, c='b', label='test dataset')
# plt.legend()
# plt.show()

# ACF, PACF 그려보기 -> p,q 구하기
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
# plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
# plt.show()

# 차분 안정성 확인 -> d 구하기
# # 1차 차분 구하기
# print('\033[31m'+'\033[1m' + "1차 차분 구하기 :" + '\033[0m')
# diff_1 = ts_log.diff(periods=1).iloc[1:]

# ARIMA 모델 훈련
from statsmodels.tsa.arima_model import ARIMA

# Build and Train  Model (p, d, q)
model = ARIMA(train_data, order=(3, 1, 0))  
fitted_m = model.fit(disp=-1)  

# 최종 예측 모델 정확도 측정(MAPE)
# print(fitted_m.summary())
# fitted_m.plot_predict()
# Forecast : 결과가 fc에 담깁니다. 
predict_num=5
start_date = pd.to_datetime('2020-03-01')
end_date = pd.to_datetime('2020-03-05') #60 2-29 / 20 1-21 / 120 4-30
dates = pd.date_range(start_date,end_date,freq='D')
fc, se, conf = fitted_m.forecast(predict_num, alpha=0.05)  # 95% conf
print(fc)

# # Make as pandas series
# fc_series = pd.Series(fc, index=test_data.index)   # 예측결과
# lower_series = pd.Series(conf[:, 0], index=test_data.index)  # 예측결과의 하한 바운드
# upper_series = pd.Series(conf[:, 1], index=test_data.index)  # 예측결과의 상한 바운드

fc_series = pd.Series(fc, index=dates)   # 예측결과
lower_series = pd.Series(conf[:, 0], index=dates)  # 예측결과의 하한 바운드
upper_series = pd.Series(conf[:, 1], index=dates)  # 예측결과의 상한 바운드

print(fc_series)
for ending_price in fc_series.values:
    print(ending_price)
#제일 나중값
print()
print(fc_series[-1])
#현재가
print(ts_log["Close"][-1])
'''
5분
363.5518599538409
362.95
'''

# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(fc_series, c='r',label='predicted price')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()

