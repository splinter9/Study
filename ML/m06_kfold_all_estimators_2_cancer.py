import numpy as np
from xgboost import XGBClassifier
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


dataset = load_breast_cancer()
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
ACC : [0.95614035 0.98245614 0.94736842 0.97368421 0.95575221] 
cross_val_score : 0.9631

AdaBoostClassifier 의 정답률 :  0.9736842105263158
BaggingClassifier 의 정답률 :  0.956140350877193
BernoulliNB 의 정답률 :  0.6228070175438597
CalibratedClassifierCV 의 정답률 :  0.9298245614035088
ComplementNB 의 정답률 :  0.9385964912280702
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.6228070175438597
ExtraTreeClassifier 의 정답률 :  0.9298245614035088
ExtraTreesClassifier 의 정답률 :  0.9736842105263158
GaussianNB 의 정답률 :  0.9736842105263158
GaussianProcessClassifier 의 정답률 :  0.9298245614035088
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.41228070175438597
LabelSpreading 의 정답률 :  0.41228070175438597
LinearDiscriminantAnalysis 의 정답률 :  0.956140350877193
LinearSVC 의 정답률 :  0.9298245614035088
LogisticRegression 의 정답률 :  0.9649122807017544
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9473684210526315
MultinomialNB 의 정답률 :  0.9385964912280702
NearestCentroid 의 정답률 :  0.9385964912280702
NuSVC 의 정답률 :  0.9122807017543859
PassiveAggressiveClassifier 의 정답률 :  0.8947368421052632
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.956140350877193
RidgeClassifierCV 의 정답률 :  0.9649122807017544
SGDClassifier 의 정답률 :  0.9473684210526315
SVC 의 정답률 :  0.9473684210526315
'''