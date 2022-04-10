from fbprophet import Prophet
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


ts_log2 = ts_log.reset_index()
ts_log2.columns = ['ds', 'y']

prophet = Prophet(yearly_seasonality=False, 
                  weekly_seasonality=False,
                  daily_seasonality='auto',
                  changepoint_range=1,
                  changepoint_prior_scale=0.05)
prophet.fit(ts_log2)

# 이후 periods 간의 데이터 예측
future = prophet.make_future_dataframe(periods=10)
forecast = prophet.predict(future)

predict_price = round(forecast[["yhat"]].iloc[-1],2).values[0]
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

prophet.plot(forecast)
plt.show()