import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
 

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
df = f1m2[::-1]


from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(df, order=(2,2,2))
# model_fit = model.fit()
# model_fit = model.fit(trend = 'c', full_output = True, disp = True)
model_fit = model.fit(trend='nc',full_output=True, disp=1)
print(model_fit.summary())

from statsmodels.tsa.stattools import acf

# 테스트 데이터 개수만큼 예측
train, test = df[:int(len(df)*0.9)], df[int(len(df)*0.9):]
# Forecast
fc, se, conf = model_fit.forecast(int(len(df)*0.1+1), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()