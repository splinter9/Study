from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler
# scalar.fit(x_train)
# x_train = scalar.transform(x_train)
# x_test = scalar.transform(x_test)

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
모델의 갯수: 41
AdaBoostClassifier 의 정답률 :  0.9166666666666666
BaggingClassifier 의 정답률 :  0.9166666666666666
BernoulliNB 의 정답률 :  0.3888888888888889
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
ComplementNB 의 정답률 :  0.6944444444444444
DecisionTreeClassifier 의 정답률 :  0.9444444444444444
DummyClassifier 의 정답률 :  0.3888888888888889
ExtraTreeClassifier 의 정답률 :  0.9722222222222222
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.3611111111111111
GradientBoostingClassifier 의 정답률 :  0.9444444444444444
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.7222222222222222
LabelPropagation 의 정답률 :  0.5
LabelSpreading 의 정답률 :  0.5
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.8888888888888888
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.2222222222222222
MultinomialNB 의 정답률 :  0.8888888888888888
NearestCentroid 의 정답률 :  0.7777777777777778
NuSVC 의 정답률 :  0.9722222222222222
PassiveAggressiveClassifier 의 정답률 :  0.4722222222222222
Perceptron 의 정답률 :  0.75
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.5555555555555556
SVC 의 정답률 :  0.8055555555555556
'''