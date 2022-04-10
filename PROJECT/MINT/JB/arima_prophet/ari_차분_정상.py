import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
 

start_date = pd.to_datetime('2020-01-01')
# end_date = pd.to_datetime('2020-02-29') #60 2-29 / 20 1-21 / 120 4-30
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

# from statsmodels.tsa.stattools import adfuller

# diff_1 = df.diff(periods=1).iloc[1:]
# print(f' p-value: {adfuller(diff_1)[1]}')

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # plot_acf(diff_1)
# # plot_pacf(diff_1)

# # plot_acf(df)
# # plot_pacf(df)

# diff_1 = df.diff(periods=1).iloc[1:]


# # plt.plot(diff_1)



# plt.show(), trend = 'nc',full_output = True, disp = True
import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df, order = (2,1,2))
model_fit = model.fit( trend = 'nc')
print(model_fit.summary())

start_date = pd.to_datetime('2021-01-31')
end_date = pd.to_datetime('2021-02-19') 
dates = pd.date_range(start_date,end_date,freq='D')

predict_num = 20

fc, se, conf = model_fit.forecast(predict_num, alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=dates)  

predict_price = round(fc_series[-1],2)
now_price = df["Close"][-1]
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
plt.plot(df, label='training')
plt.plot(fc_series, c='r',label='predicted price')
plt.legend()
plt.show()






