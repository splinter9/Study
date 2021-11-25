#1. R2를 음수가 아닌 0.5이하로 만들것
#2. 데이터는 손대지 말것
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#7. train 70%



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.7, shuffle=True, random_state=65)

print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=101, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  #로스값은 훈련에 영향을 주지 않는다, 결과니까...
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

#plt.scatter(x,y)  #점찍기
#plt.plot(x, y_predict) #선긋기
#plt.show()
