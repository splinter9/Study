import numpy as np

a = np.array(range(1,11))                     ## [1~10]
size = 5                                      ## size는 5 라는 숫자로 정한다

def split_x(dataset, size):                   ## split_x는 dataset부터 size 크기로 배열한다
    aaa = []                                  ## aaa는 리스트다
    for i in range(len(dataset) - size + 1):  ## for i in range 반복한다 dataset갯수에 뺀다 (size크기에 +1)해서
        subset = dataset[i : (i + size)]       ## subset은 dataset을 리스트로 만드는데 for 시작부너 시작에 사이즈를 더해준값을 잘라낸다
        aaa.append(subset)                    ## aaa리스트에 돌아간 dataset을 추가해준다
    return np.array(aaa)                      ## 이렇게 만들어진 aaa리스트를 넌파이를 적용한다

dataset = split_x(a, size)                    ## dataset은 split_x를 a부터 size 크기로 배열한다

print(dataset)                                ## dataset을 출력한다

bbb = split_x(a, size)                        ## bbb는 split_x를 a부터 size 크기로 배열한다
print(bbb)                                    ## bbb를 출력한다
print(bbb.shape)

x = bbb[:, :4]
y = bbb[:, 4]     
print(x,y)
print(x.shape, y.shape)
