# print문과 format함수


a = '사과'
b = '배'
c = '옥수수'


print(a, b, c)
print('나는 {0}을 먹었다'.format(a))
print('나는 {0}와 {1}을 먹었다'.format(a,b))
print('나는 {0}와 {1}와 {2}를 먹었다'.format(a,b,c))


print('나는 ', a,'를 먹었다.', sep='')
print('나는 ', a,'와 ',b,'를 먹었다.', sep='')
print('나는 ', a,'와 ',b,'와',c,'를 먹었다.', sep='') # ,로 묶으면 띄어쓰기가 써지나보넹
print('나는 ', a,'와 ',b,'와',c,'를 먹었다.', sep='#') # sep='' 파라미터는 ,사이 구분자기호 표현하는건가 보다


print('나는 %s와 %s와 %s를 먹었다'%(a,b,c))
