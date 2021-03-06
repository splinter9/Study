from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 데이터 정제작업 !!
datasets = load_breast_cancer()
x = datasets.data           # 원본데이터는 나중에 어떻게 쓸지 모르니까  그냥 둠
y = datasets.target         

# print(x.shape) # x형태  (##,##)     4차원 형태로 conv2d로 받아서 cnn모델링가능
# print(y.shape) # y형태  (###, )
# print(datasets.feature_names) # 컬럼,열의 이름들
# print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명


# cnn 만들기
# img           (1000, 32,32,3) -> (1000, 3072) -> 4차원에서 2차원으로변환  dnn
# 2차원         (1000, 3072)  -> (1000, 32,32,3) -> 2차원에서 4차원으로 형태변환 한후 conv2D 하다가 다시 flatten써서 dnn

# numpy pandas로 변환후 pandas의 제공기능인 index정보와 columns정보를 확인할수있다.
xx = pd.DataFrame(x, columns=datasets.feature_names)    # x가 pandas로 바껴서 xx에 저장, columns를 칼럼명이 나오게 지정해준다.
#print(type(xx))         # pandas.core.frame.DataFrame
#print(xx)               # 잘 되었나 확인. index와 colmuns의 이름이 나옴.

#print(xx.corr())        # 칼럼들의 서로서로의 상관관게를 수치로 확인할 수 있다.    절대값클수록 양 or 음의 상관관계 0에 가까울수록 서로 영향 없음

#xx['price'] = y         # xx의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 원본데이터는 그대로있다.    열 추가하는 방법.

#print(xx)              # price열이 추가되어 있는 것 확인.

# print(xx.corr())      # price와 어떤 열이 제일 상관관계가 적은지 확인.

#########################################################
# import matplotlib.pyplot as plt
# import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리
# plt.show()
###########################################################
#xx= xx.drop(['CHAS','price'], axis=1)    # x데이터에서 CHAS열 제거
#print(xx)     #CHAS열이 제거되고 12개의 columns가 있는것 확인.

xx = xx.values     #xx = xx.to_numpy()  2가지방법중 아무거나 사용해서 xx를 다시 numpy로 바꿔준다. 
#왜냐하면 차원변환해줘야하는데 numpy만 된다하네


x_train,x_test,y_train,y_test = train_test_split(xx,y, train_size=0.7, shuffle=True, random_state=49)

#xx -> x_train과 x_test로 분리.  y-> y_train과 y_test로 분리되었다.

print(x_train.shape,y_train.shape)      #(398, 30) (398,)
print(x_test.shape,y_test.shape)        #(171, 30) (171,)

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()     
#스케일러를 쓸려면 또 여기단계에서 써줘야한다. 2차원일때 아 귀찮아 ㅋㅋㅋㅋ

x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,5,3)
x_test = scaler.transform(x_test).reshape(len(x_test),2,5,3)

#print(x_train[:3])     x데이터에 스케일러값이 잘 들어갔나 확인.
print(x_train.shape)    # (309, 2, 5, 1)
print(x_test.shape)     # (133, 2, 5, 1)


#2.모델링

model = Sequential()
model.add(Conv2D(100,kernel_size=(2,2),strides=1,padding='same', input_shape=(2,5,3), activation='relu'))    # 2,2,10                                                                           # 1,1,10
model.add(Conv2D(200,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 2,2,10 
model.add(MaxPooling2D(2,2))                                                                                # 1,1,10     
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_1_boston{krtime}.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=13,validation_split=0.3, callbacks=[es])#,mcp

#model.save(f"./_save/keras35_1_boston{krtime}.h5")

#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)



