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
train, test = ts_log2[:int(len(ts_log2)*0.9)], ts_log2[int(len(ts_log2)*0.9):]

prophet = Prophet(yearly_seasonality=False, 
                  weekly_seasonality=False,
                  daily_seasonality='auto',
                  changepoint_range=1,
                  changepoint_prior_scale=0.05)
prophet.fit(train)
# 3) 예측할 기간 입력
pred_date = prophet.make_future_dataframe(periods=int(len(test)), freq='D')
# 4) predict
predictions = prophet.predict(pred_date)

# changepoint 시각화
from fbprophet.plot import add_changepoints_to_plot
fig = prophet.plot(predictions)
a = add_changepoints_to_plot(fig.gca(), prophet, predictions) 


predictions.plot(x='ds', y='yhat', label='prediction value', legend=True, figsize=(12,8))
plt.show()


# test.plot(x='ds', y='y', label='actual value', legend=True, figsize=(12,8))
# plt.show()
from statsmodels.tools.eval_measures import rmse


from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

initial = 300 # initial : train을 수행할 기간
initial = str(initial) + ' days'

horizon = 30 # horizon : 예측할 기간
horizon = str(horizon) + ' days'

period = 60 # period : cutoff date 간 간격
period = str(period) + ' days'

result_cv = cross_validation(prophet, initial=initial, period=period, horizon=horizon)
print(performance_metrics(result_cv).tail(10))

plot_cross_validation_metric(result_cv, metric='rmse')
plt.show()
print(rmse(predictions.iloc[-1*len(test):]['yhat'], test['y']))