# 자료형
#1. 리스트

a = [1,2,3,4,5]
b = [1,2,3,'a','b']
print(b)

# 넘파이는 딱 한가지 자료형 nd.array 얘만 들어갈 수 있다. / 리스트는 다 들어갈 수 있음

# 슬라이싱이 사실상 str보다는 list꺼징...

print(a[0] + a[3])  # 5
print(str(b[0]) + b[3])  # 1a
print(type(a))  # <class 'list'>
print(a[-2])  # 4
print(a[1:3])  # [2, 3]

a = [1, 2, 3, ['a', 'b', 'c']]
print(a[2])  # 3
print(a[3])  # ['a', 'b', 'c']
print(a[3][0])  # a
print(a[-1][0])  # a
print(a[:2])  # [1, 2]

a = [1,2,3]
b = [4,5,6]
c = [7,8,9,10]
print(a+b)  # [1, 2, 3, 4, 5, 6] -> 그래서 인공지능 수식계산은 numpy.ndarray 로 하고 있음
print(a+c) # [1, 2, 3, 7, 8, 9, 10]
print(a*3) # [1, 2, 3, 1, 2, 3, 1, 2, 3]
print(str(a[0])+'히잉')

print(a[2]+5)

# 리스트 관련 함수 (.append/ .sort()/ .reverse()/ .insert(자리,인자), .remove(인자), 걍 '적용'이 디폴트인듯 )
a = [1,2,3]
a.append(4)
print(a)

a = [1,3,4,2]
a.sort() # 걍 아예 바꿔버리네?
print(a) # [1, 2, 3, 4]

a.reverse() # [4, 3, 2, 1]
print(a)
print(a.index(3)) # 1

a.insert(0, 7.1) # 와우 일케쓰네?/ 0자리에 7.1 넣는다. '교체하다' 아니다./ 그 나머지는 뒤로 밀린다
print(a)
a.insert(3, 3)
print(a)
a[5] = 1000 # 아예 교체해버림. 근데 보통 교체해버리지 넣고 뒤로 미는 경우 없기 때문에 이 케이스를 주로 쓴다.
print(a)

a.remove(7.1) # a라는 list에서 7.1을 지워라/ 인자 자체를 넣어준다/두개 있을때 먼저 있는애부터 지워준다.
print(a)

