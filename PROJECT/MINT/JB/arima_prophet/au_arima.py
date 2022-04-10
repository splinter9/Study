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


from pmdarima.arima import ndiffs
import pmdarima as pm

kpss_diffs = ndiffs(ts_log, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(ts_log, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)
# print(f"추정된 차수 d = {n_diffs}")
best_model = pm.auto_arima(y = ts_log    
                      , d = n_diffs            
                      , start_p = 0 #(기본값 = 2), max_p (기본값 = 5): AR(p)를 찾을 범위 (start_p에서 max_p까지 찾는다!)
                      , max_p = 3   
                      , start_q = 0 # (기본값 = 2), max_q (기본값 = 5): AR(q)를 찾을 범위 (start_q에서 max_q까지 찾는다!)
                      , max_q = 3   
                      , m = 1  #(기본값 = 1): 계절적 차분이 필요할 때 쓸 수 있는 모수로 
                               # m=4이면 분기별, m=12면 월별, m=1이면 계절적 특징을 띠지 않는 데이터를 의미
                               # m=1이면 자동적으로 seasonal 에 대한 옵션은 False로 지정.     
                      , seasonal = False #(기본값 = True): 계절성 ARIMA 모형을 적합할지의 여부
                      , stepwise = False #(기본값 = True): 최적의 모수를 찾기 위해 쓰는 힌드만 - 칸다카르 알고리즘을 사용할지의 여부
                                        #False면 모든 모수 조합으로 모형을 적합한다.
                      , trace=True #(기본값 = False): stepwise로 모델을 적합할 때마다 결과를 프린트하고 싶을 때 사용한다.
                      )
print("best model --> (p, d, q):", best_model.order)
print(best_model.summary())

# best_model.plot_diagnostics(figsize=(16, 8))
# plt.show()

# 테스트 데이터 개수만큼 예측
train_data, test_data = ts_log[:int(len(ts_log)*0.7)], ts_log[int(len(ts_log)*0.7):]

y_predict = best_model.predict(n_periods=len(test_data)) 
y_predict = pd.DataFrame(y_predict,index = test_data.index,columns=['Prediction'])

# 그래프
# fig, axes = plt.subplots(1, 1, figsize=(12, 4))
# plt.plot(train_data, label='Train')        # 훈련 데이터
# plt.plot(test_data, label='Test')          # 테스트 데이터
# plt.plot(y_predict, label='Prediction')  # 예측 데이터
# plt.legend()
# plt.show()

def forecast_one_step():
    fc, conf_int = best_model.predict(n_periods=1 # 한 스텝씩!
        , return_conf_int=True)              # 신뢰구간 출력
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0]
    )
    
forecasts = []
y_pred = []
pred_upper = []
pred_lower = []

for new_ob in test_data:
    fc, conf = forecast_one_step()
    y_pred.append(fc)
    pred_upper.append(conf[1])
    pred_lower.append(conf[0])

    ## 모형 업데이트 !!
    best_model.update(new_ob)
    
print(pd.DataFrame({"test": test_data, "pred": y_pred}))

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = go.Figure([
    # 훈련 데이터-------------------------------------------------------
    go.Scatter(x = train_data.index, y = train_data, name = "Train", mode = 'lines'
              ,line=dict(color = 'royalblue'))
    # 테스트 데이터------------------------------------------------------
    , go.Scatter(x = test_data.index, y = test_data, name = "Test", mode = 'lines'
                ,line = dict(color = 'rgba(0,0,30,0.5)'))
    # 예측값-----------------------------------------------------------
    , go.Scatter(x = test_data.index, y = y_pred, name = "Prediction", mode = 'lines'
                     ,line = dict(color = 'red', dash = 'dot', width=3))
    
    # 신뢰 구간---------------------------------------------------------
    , go.Scatter(x = test_data.index.tolist() + test_data.index[::-1].tolist() 
                ,y = pred_upper + pred_lower[::-1] ## 상위 신뢰 구간 -> 하위 신뢰 구간 역순으로
                ,fill='toself'
                ,fillcolor='rgba(0,0,30,0.1)'
                ,line=dict(color='rgba(0,0,0,0)')
                ,hoverinfo="skip"
                ,showlegend=False)
])

fig.update_layout(height=400, width=1000, title_text="ARIMA(0,1,0)모형")
fig.show()