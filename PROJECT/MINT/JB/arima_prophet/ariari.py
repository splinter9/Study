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

# plt.figure(figsize=(13,6))
# plt.plot(df)
# plt.show()

# from numpy import log


from statsmodels.tsa.stattools import adfuller
print(f' p-value: {adfuller(df)[1]}')
'''
ADF 검정의 귀무 가설 : 시계열이 비정상적
검정의 p-값이 유의 수준(0.05)보다 작으면 귀무 가설을 기각
시계열이 실제로 정상적이라고 추론
p-value: 0.392837
'''
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# (AR = 2, 차분 =1, MA=2) 파라미터로 ARIMA 모델을 학습한다.
model = ARIMA(df, order = (2,2,0))
model_fit = model.fit(trend = 'c', full_output = True, disp = True)
print(model_fit.summary())
fig = model_fit.plot_predict()
plt.show()
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title = "실제값과 예측값의 잔차")
plt.show()
