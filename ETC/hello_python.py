
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML 

#세타 초기값
#상,우,하,좌
theta_0 = np.array([
                   [np.nan,1,1,np.nan], #0
                   [np.nan,1,1,1],#1
                   [np.nan,np.nan,np.nan,1],#2
                   [1,np.nan,1,np.nan],#3
                   [1,1,np.nan,np.nan],#4
                   [np.nan,np.nan,1,1],#5
                   [1,1,np.nan,np.nan],#6
                   [np.nan,np.nan,np.nan,1]])#7
#세타를 정책으로 변환
def get_pi(theta):
    #비율계산
    [m,n]=theta.shape
    pi =np.zeros((m,n))
    for i in range(0,m):
        pi[i,:] =theta[i,:] / np.nansum(theta[i,:])
    pi = np.nan_to_num(pi)
    return pi

#세타의 초기값을 정책으로 변환
pi_0 = get_pi(theta_0)
print(pi_0)

def get_s_next(s,a):
    if a==0:#상
        return s-3 
    elif a ==1:#우
        return s+1
    elif a==2: #하
        return s+3
    elif a==3: #좌
        return s-1 

#헁똥
[a, b] = theta_0.shape
Q = np.random.rand(a,b) * theta_0
print(Q)