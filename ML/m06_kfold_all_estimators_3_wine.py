import numpy as np
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


#2.MODEL
model = XGBClassifier()

scores = cross_val_score(model, x, y, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))


#2. 모델 구성
allAlgorithms = all_estimators(type_filter = 'classifier')
# allAlgorithms = all_estimators(type_filter = 'regressor')  
# allAlgorithms XGBoost, Catboost, LGBM은 없다. >> 
print('allAlgorithms :', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms))

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name,'은 없는 모델')

'''
ACC : [0.97222222 0.97222222 0.97222222 0.91428571 1.        ] 
cross_val_score : 0.9662

AdaBoostClassifier 의 정답률 :  0.9166666666666666
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.3888888888888889
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.6944444444444444
DecisionTreeClassifier 의 정답률 :  0.9444444444444444
DummyClassifier 의 정답률 :  0.3888888888888889
ExtraTreeClassifier 의 정답률 :  0.7222222222222222
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.3611111111111111
GradientBoostingClassifier 의 정답률 :  0.9444444444444444
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.7222222222222222
LabelPropagation 의 정답률 :  0.5
LabelSpreading 의 정답률 :  0.5
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.6944444444444444
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.8888888888888888
NearestCentroid 의 정답률 :  0.7777777777777778
NuSVC 의 정답률 :  0.9722222222222222
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.5833333333333334
Perceptron 의 정답률 :  0.75
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.5833333333333334
SVC 의 정답률 :  0.8055555555555556
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''


