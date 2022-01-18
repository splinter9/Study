import numpy as np
from xgboost import XGBClassifier
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


dataset = load_iris()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)


model = XGBClassifier()
scores = cross_val_score(model, x, y, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))

#2.MODEL
allAlgorithms = all_estimators(type_filter='classifier')

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수:", len(allAlgorithms)) #41

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        continue
        #print(name,'없는 모델')

'''
ACC : [1.         0.96666667 0.93333333 0.93333333 0.93333333] 
cross_val_score : 0.9533

AdaBoostClassifier 의 정답률 :  1.0
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9666666666666667
CategoricalNB 의 정답률 :  0.9666666666666667
ComplementNB 의 정답률 :  0.7
DecisionTreeClassifier 의 정답률 :  1.0
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  1.0
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  1.0
GradientBoostingClassifier 의 정답률 :  1.0
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  1.0
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  0.9
NearestCentroid 의 정답률 :  0.9666666666666667
NuSVC 의 정답률 :  1.0
PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
Perceptron 의 정답률 :  0.8
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 :  1.0
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9
RidgeClassifierCV 의 정답률 :  0.9
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  1.0
'''