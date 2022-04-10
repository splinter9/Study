import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2020-02-29') #60 2-29 / 20 1-21 / 120 4-30
end_date = pd.to_datetime('2021-01-30') #60 2-29 / 20 1-21 / 120 4-30 / 396 2021-1-31
period = 396

dates = pd.date_range(start_date,end_date,freq='D')[::-1]

f_1m = pd.read_csv("CF 202206_1M.csv")
f1m = pd.DataFrame()
# print(f_1m.iloc[period-1]) # 09:00:00 확인

f1m["Close"] = f_1m["종가"]
# f1m2 = f1m.loc[0:59]
f1m2 = f1m.loc[0:period-1]

# print(f1m2.head())
f1m2.index = dates
f1m2.index.name = "day"
ts_log = f1m2[::-1]
# train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택

# 하루치 1분봉 시계열 종가 데이터 준비 
train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]

# plt.figure(figsize=(13,6))
# plt.grid(True)
# plt.plot(ts_log, c='r', label='training dataset')  
# plt.plot(test_data, c='b', label='test dataset')
# plt.legend()
# plt.show()

# ACF, PACF 그려보기 -> p,q 구하기
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# plt.plot(ts_log)
# plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
# plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
# plt.show()

# 차분 안정성 확인 -> d 구하기
# 1차 차분 구하기
print('\033[31m'+'\033[1m' + "1차 차분 구하기 :" + '\033[0m')
diff_1 = ts_log.diff(periods=1).iloc[1:]
plt.plot(diff_1)
plt.show()
plot_acf(diff_1)   # ACF : Autocorrelation 그래프 그리기
plot_pacf(diff_1)  # PACF : Partial Autocorrelation 그래프 그리기



# ARIMA 모델 훈련
from statsmodels.tsa.arima_model import ARIMA
# Build and Train  Model (p, d, q)
model = ARIMA(train_data, order=(0, 1, 1))  
fitted_m = model.fit(trend='nc',full_output=True, disp=True)  #상수항 사용 X
'''
disp : bool, optional
If True, convergence information is printed. 
For the default l_bfgs_b solver,
disp controls the frequency of the output during the iterations. 
disp < 0 means no output in this case.
'''
print(fitted_m.summary())

# 최종 예측 모델 정확도 측정(MAPE)





fitted_m.plot_predict()
# Forecast : 결과가 fc에 담깁니다. 
print(len(test_data))
fc, se, conf = fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)   # 예측결과
lower_series = pd.Series(conf[:, 0], index=test_data.index)  # 예측결과의 하한 바운드
upper_series = pd.Series(conf[:, 1], index=test_data.index)  # 예측결과의 상한 바운드

#모델의 오차율 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = mean_squared_error(test_data, fc)
print('\033[31m'+'\033[1m' + 'MSE: ' + '\033[0m', mse)

mae = mean_absolute_error(test_data, fc)

print('\033[31m'+'\033[1m' + 'MAE: ' + '\033[0m', mae)

rmse = math.sqrt(mean_squared_error(test_data, fc))

print('\033[31m'+'\033[1m' + 'RMSE: ' + '\033[0m', rmse)

# # mape = np.mean(np.abs((fc) - (test_data))/np.abs((test_data)))
# mape = np.mean(np.abs((test_data - fc) / test_data)) * 100 
# print('\033[31m'+'\033[1m' + 'MAPE: ' + '\033[0m'+ '{:.2f}%'.format(mape))


# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()

