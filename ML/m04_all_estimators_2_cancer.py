from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


dataset = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 :  0.9736842105263158
BaggingClassifier 의 정답률 :  0.9473684210526315
BernoulliNB 의 정답률 :  0.6228070175438597
CalibratedClassifierCV 의 정답률 :  0.9298245614035088
ComplementNB 의 정답률 :  0.9385964912280702
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.6228070175438597
ExtraTreeClassifier 의 정답률 :  0.9385964912280702
ExtraTreesClassifier 의 정답률 :  0.9736842105263158
GaussianNB 의 정답률 :  0.9736842105263158
GaussianProcessClassifier 의 정답률 :  0.9298245614035088
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.41228070175438597
LabelSpreading 의 정답률 :  0.41228070175438597
LinearDiscriminantAnalysis 의 정답률 :  0.956140350877193
LinearSVC 의 정답률 :  0.8333333333333334
LogisticRegression 의 정답률 :  0.9649122807017544
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9649122807017544
MultinomialNB 의 정답률 :  0.9385964912280702
NearestCentroid 의 정답률 :  0.9385964912280702
NuSVC 의 정답률 :  0.9122807017543859
PassiveAggressiveClassifier 의 정답률 :  0.9122807017543859
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.956140350877193
RidgeClassifierCV 의 정답률 :  0.9649122807017544
SGDClassifier 의 정답률 :  0.5964912280701754
SVC 의 정답률 :  0.9473684210526315
'''


