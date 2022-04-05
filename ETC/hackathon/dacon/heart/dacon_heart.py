from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches


path = "../_data/dacon/heart/"
train = pd.read_csv(path + 'train.csv')
print(train.shape) #(151, 15)
test_file = pd.read_csv(path + 'test.csv')
submit_file = pd.read_csv(path + 'sample_submission.csv')  

'''
def check_missing_col(dataframe):
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'총 {missing_values}개의 결측치가 존재합니다.')

        if i == len(dataframe.columns) - 1 and counted_missing_col == 0:
            print('결측치가 존재하지 않습니다') #결측치가 존재하지 않습니다

check_missing_col(train)

print(train.describe())
#          id         age         sex          cp    trestbps        chol         fbs     restecg     thalach       exang     oldpeak       slope          ca        thal      target
# count  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000  151.000000
# mean    76.000000   54.496689    0.682119    1.066225  132.033113  244.529801    0.158940    0.509934  150.629139    0.324503    0.976821    1.377483    0.602649    2.317881    0.549669
# std     43.734045    8.904586    0.467202    1.056213   17.909929   56.332206    0.366837    0.514685   23.466463    0.469747    1.085998    0.640226    0.917093    0.604107    0.499183
# min      1.000000   34.000000    0.000000    0.000000   94.000000  131.000000    0.000000    0.000000   88.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
# 25%     38.500000   48.500000    0.000000    0.000000  120.000000  208.000000    0.000000    0.000000  136.500000    0.000000    0.000000    1.000000    0.000000    2.000000    0.000000
# 50%     76.000000   54.000000    1.000000    1.000000  130.000000  239.000000    0.000000    1.000000  155.000000    0.000000    0.800000    1.000000    0.000000    2.000000    1.000000
# 75%    113.500000   61.000000    1.000000    2.000000  140.000000  270.000000    0.000000    1.000000  168.000000    1.000000    1.600000    2.000000    1.000000    3.000000    1.000000
# max    151.000000   77.000000    1.000000    3.000000  192.000000  564.000000    1.000000    2.000000  195.000000    1.000000    5.600000    2.000000    3.000000    3.000000    1.000000

plt.style.use("ggplot")
# 히스토그램 을 사용해서 데이터의 분포를 살펴봅니다.
plt.figure(figsize=(25,20))
plt.suptitle("Data Histogram", fontsize=40)

# id는 제외하고 시각화합니다.
cols = train.columns[1:]
for i in range(len(cols)):
    plt.subplot(5,3,i+1)
    plt.title(cols[i], fontsize=20)
    if len(train[cols[i]].unique()) > 20:
        plt.hist(train[cols[i]], bins=20, color='b', alpha=0.7)
    else:
        temp = train[cols[i]].value_counts()
        plt.bar(temp.keys(), temp.values, width=0.5, alpha=0.7)
        plt.xticks(temp.keys())
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, axes = plt.subplots(5, 3, figsize=(25, 20))

fig.suptitle('feature distributions per quality', fontsize= 40)
for ax, col in zip(axes.flat, train.columns[1:-1]):
    sns.violinplot(x= 'target', y= col, ax=ax, data=train)
    ax.set_title(col, fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.figure(figsize=(20,10))

heat_table = train.drop(['id'], axis=1).corr()
heatmap_ax = sns.heatmap(heat_table, annot=True, cmap='coolwarm')
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=15, rotation=45)
heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=15)
plt.title('correlation between features', fontsize=40)
plt.show()
'''

x = train.iloc[:, 1:-1]
y = train.iloc[:, -1]
test_file = test_file.drop(columns=['id'], axis=1)


print(x)
print(y)
print(x.shape)  #(151, 15) dim=15
print(y.shape)  #(151,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size=0.8, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=14))
model.add(Dense(80))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16, 
          #validation_data=(x_val, y_val))
          validation_split=0.1)

#validation_split을 사용하면 굳이 위에 데이터 정제작업에서 스플릿안해도 된다

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn import metrics

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score


############ 제출용 제작 ##############
results = model.predict(test_file)
submit_file['target'] = results
submit_file.to_csv(path+f"results.csv", index = False)
