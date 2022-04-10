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
ss = f1m2[::-1]
# print(ss.tail())
'''
             Close
day
2020-01-01  363.30
2020-01-02  363.40
2020-01-03  363.45
2020-01-04  363.45
2020-01-05  363.55
'''

from pmdarima.arima import ndiffs
import pmdarima as pm

y_train = ss['Close'][:int(0.7*len(ss))]
y_test = ss['Close'][int(0.7*len(ss)):]
# y_train.plot()
# y_test.plot()
# plt.show()

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"추정된 차수 d = {n_diffs}")
# n_diffs = 1

model = pm.auto_arima(y = y_train        # 데이터
                      , d = n_diffs            # 차분 차수, ndiffs 결과!
                      , start_p = 0 
                      , max_p = 3   
                      , start_q = 0 
                      , max_q = 3   
                      , m = 1       
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True
                      , trace=True
                      )

model = pm.auto_arima (y_train, d = n_diffs, seasonal = False, trace = False)
model_fit = model.fit(y_train)
'''
추정된 차수 d = 2
Performing stepwise search to minimize aic
 ARIMA(0,2,0)(0,0,0)[0] intercept   : AIC=-65.243, Time=0.03 sec
 ARIMA(1,2,0)(0,0,0)[0] intercept   : AIC=-80.159, Time=0.03 sec
 ARIMA(0,2,1)(0,0,0)[0] intercept   : AIC=-91.981, Time=0.11 sec
 ARIMA(0,2,0)(0,0,0)[0]             : AIC=-67.219, Time=0.02 sec
 ARIMA(1,2,1)(0,0,0)[0] intercept   : AIC=-93.349, Time=0.10 sec
 ARIMA(2,2,1)(0,0,0)[0] intercept   : AIC=-94.554, Time=0.12 sec
 ARIMA(2,2,0)(0,0,0)[0] intercept   : AIC=-93.825, Time=0.08 sec
 ARIMA(3,2,1)(0,0,0)[0] intercept   : AIC=-92.811, Time=0.17 sec
 ARIMA(2,2,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec
 ARIMA(1,2,2)(0,0,0)[0] intercept   : AIC=-94.294, Time=0.22 sec
 ARIMA(3,2,0)(0,0,0)[0] intercept   : AIC=-93.106, Time=0.10 sec
 ARIMA(3,2,2)(0,0,0)[0] intercept   : AIC=-94.676, Time=0.25 sec
 ARIMA(3,2,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.22 sec
 ARIMA(2,2,3)(0,0,0)[0] intercept   : AIC=-95.444, Time=0.25 sec
 ARIMA(1,2,3)(0,0,0)[0] intercept   : AIC=-97.110, Time=0.20 sec
 ARIMA(0,2,3)(0,0,0)[0] intercept   : AIC=-94.132, Time=0.17 sec
 ARIMA(0,2,2)(0,0,0)[0] intercept   : AIC=-95.491, Time=0.10 sec
 ARIMA(1,2,3)(0,0,0)[0]             : AIC=inf, Time=0.18 sec

Best model:  ARIMA(1,2,3)(0,0,0)[0] intercept
Total fit time: 2.554 seconds
'''
#Best model:  ARIMA(1,2,3)(0,0,0)[0] intercept
print(model.summary())
'''
                               SARIMAX Results
==============================================================================
Dep. Variable:                      y   No. Observations:                   42
Model:               SARIMAX(1, 2, 3)   Log Likelihood                  54.555
Date:                Sun, 03 Apr 2022   AIC                            -97.110
Time:                        04:26:15   BIC                            -86.977
Sample:                             0   HQIC                           -93.446
                                 - 42
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept     -0.0006      0.005     -0.130      0.897      -0.010       0.009
ar.L1         -0.6284      0.281     -2.237      0.025      -1.179      -0.078
ma.L1         -0.3370      0.565     -0.596      0.551      -1.445       0.771
ma.L2         -0.7362      0.642     -1.146      0.252      -1.995       0.523
ma.L3          0.5634      0.266      2.120      0.034       0.042       1.084
sigma2         0.0035      0.002      1.853      0.064      -0.000       0.007
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):                 1.39
Prob(Q):                              0.88   Prob(JB):                         0.50
Heteroskedasticity (H):               1.83   Skew:                             0.36
Prob(H) (two-sided):                  0.29   Kurtosis:                         2.43
===================================================================================
Ljung-Box (Q) 융-박스 검정 통계량는 잔차가 백색잡음인지 검정한 통계량입니다.

Prob (Q) 값을 보면 0.88이므로 유의수준 0.05에서 귀무가설을 기각하지 못합니다. 
Ljung-Box (Q) 통계량의 귀무가설은 “잔차(residual)가 백색잡음(white noise) 시계열을 따른다”이므로, 
위 결과를 통해 시계열 모형이 잘 적합되었고 남은 잔차는 더이상 자기상관을 가지지 않는 백색 잡음임을 확인할 수 있습니다.

Jarque-Bera (JB) 자크-베라 검정 통계량은 잔차가 정규성을 띠는지 검정한 통계량입니다.
Prob(JB)값을 보면 0.50으로 유의 수준 0.05에서 귀무가설을 못합니다. 
Jarque-Bera (JB) 통계량의 귀무가설은 “잔차가 정규성을 만족한다”이므로, 
위 결과를 통해 “잔차가 정규성을 따른다”을 확인할 수 있습니다.

Heteroskedasticity (H) 이분산성 검정 통계량은 잔차가 이분산을 띠지 않는지 검정한 통계량입니다.

잔차가 정규분포를 따른다면, 경험적으로

비대칭도 (Skew)는 0에 가까워야 하고
첨도 (Kurtosis)는 3에 가까워야 합니다
'''
# model.plot_diagnostics(figsize=(16, 8))
# plt.show()

# 테스트 데이터 개수만큼 예측
y_predict = model.predict(n_periods=len(y_test)) 
y_predict = pd.DataFrame(y_predict,index = y_test.index,columns=['Prediction'])

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1 # 한 스텝씩!
        , return_conf_int=True)              # 신뢰구간 출력
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0]
    )
    
forecasts = []
y_pred = []
pred_upper = []
pred_lower = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    y_pred.append(fc)
    pred_upper.append(conf[1])
    pred_lower.append(conf[0])

    ## 모형 업데이트 !!
    model.update(new_ob)

# 그래프/
# fig, axes = plt.subplots(1, 1, figsize=(12, 4))
# plt.plot(y_train, label='Train')        # 훈련 데이터
# plt.plot(y_test, label='Test')          # 테스트 데이터
# plt.plot(y_predict, label='Prediction')  # 예측 데이터
# plt.legend()
# plt.show()

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = go.Figure([
    # 훈련 데이터-------------------------------------------------------
    go.Scatter(x = y_train.index, y = y_train, name = "Train", mode = 'lines'
              ,line=dict(color = 'royalblue'))
    # 테스트 데이터------------------------------------------------------
    , go.Scatter(x = y_test.index, y = y_test, name = "Test", mode = 'lines'
                ,line = dict(color = 'rgba(0,0,30,0.5)'))
    # 예측값-----------------------------------------------------------
    , go.Scatter(x = y_test.index, y = y_pred, name = "Prediction", mode = 'lines'
                     ,line = dict(color = 'red', dash = 'dot', width=3))
    
    # 신뢰 구간---------------------------------------------------------
    , go.Scatter(x = y_test.index.tolist() + y_test.index[::-1].tolist() 
                ,y = pred_upper + pred_lower[::-1] ## 상위 신뢰 구간 -> 하위 신뢰 구간 역순으로
                ,fill='toself'
                ,fillcolor='rgba(0,0,30,0.1)'
                ,line=dict(color='rgba(0,0,0,0)')
                ,hoverinfo="skip"
                ,showlegend=False)
])
def MAPE(y_test, y_pred):
    	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 
    
print(f"MAPE: {MAPE(y_test, y_pred):.3f}") #MAPE: 0.024

# fig.update_layout(height=400, width=1000, title_text="ARIMA(1,2,3)모형")
# fig.show()
forecast_data = model_fit.forecast(steps=5) 
pred_arima_y = forecast_data[0].tolist()
print(pred_arima_y)