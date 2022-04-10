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

train, test = df[:int(len(df)*0.9)], df[int(len(df)*0.9):]

from pmdarima.arima import ndiffs
import pmdarima as pm
kpss_diffs = ndiffs(train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"추정된 차수 d = {n_diffs}")#추정된 차수 d = 1
model = pm.auto_arima(y = train        # 데이터
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

model = pm.auto_arima (train, d = n_diffs, seasonal = False, trace = False)



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

for new_ob in range(1,len(test)-1):
    fc, conf = forecast_one_step()
    y_pred.append(fc)
    pred_upper.append(conf[1])
    pred_lower.append(conf[0])
    print(test.iloc[new_ob])

    ## 모형 업데이트 !!
    model.update(test.iloc[new_ob])
    
model_fit = model.fit(train)
print(model.summary())
y_predict = model.predict(n_periods=len(test)) 
y_predict = pd.DataFrame(y_predict,index = test.index,columns=['Prediction'])
    
# # 그래프/
fig, axes = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(train, label='Train')        # 훈련 데이터
plt.plot(test, label='Test')          # 테스트 데이터
plt.plot(y_predict, label='Prediction')  # 예측 데이터
plt.legend()
plt.show()