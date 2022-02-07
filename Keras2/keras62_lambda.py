# 람다함수란? 익명 함수(匿名函數, Anonymous functions)를 지칭하는 용어
# 람다식은 주로 고차 함수에 인자(argument)로 전달되거나 고차 함수가 돌려주는 결과값으로 쓰인다.

from numpy import gradient


gradient = lambda x: 2*x - 4 #

x = 3

print(gradient(x))  # 함수를 정의하고 인풋을 x로 입력한다 

def gradient2(x):
    return 2*x - 4

x = 3

print(gradient(x))
print(gradient2(x))



##### 전통적인 방법 #######
for i in range(10):
    print(i)

##### 다 나은 방법 #######
map(lambda x: print(x), range(0, 10))


## lambda는 길게 쓰는 함수를 한줄로 줄인 프로그래밍 함수이다

