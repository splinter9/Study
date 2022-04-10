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

# train_data, test_data = ts_log[:int(len(ts_log)*0.7)], ts_log[int(len(ts_log)*0.7):] 20일 데이터
start_date = pd.to_datetime('2021-01-31')
end_date = pd.to_datetime('2021-02-19') 
dates = pd.date_range(start_date,end_date,freq='D')

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(1, 1, 1))  
fitted_m = model.fit(trend='nc')  
print(fitted_m.summary())
predict_num=20

fc, se, conf = fitted_m.forecast(predict_num, alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=dates)   # 예측결과
lower_series = pd.Series(conf[:, 0], index=dates)  # 예측결과의 하한 바운드
upper_series = pd.Series(conf[:, 1], index=dates)  # 예측결과의 상한 바운드

predict_price = round(fc_series[-1],2)
now_price = ts_log["Close"][-1]
range = round(abs(now_price - predict_price),2)
print("최종예측값")
print(predict_price)
print("현재가")
print(now_price)

if now_price > predict_price:
    print(range, " 포인트 하락 예상")
else:
    print(range, " 포인트 상승 예상")
    

plt.figure(figsize=(10,5), dpi=100)
plt.plot(ts_log, label='training')
plt.plot(fc_series, c='r',label='predicted price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()

# print(fc_series)
# for ending_price in fc_series.values:
#     print(ending_price)
#제일 나중값

# Plot
