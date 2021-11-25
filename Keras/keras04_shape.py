import numpy as np
print(np.__version__)


a1=np.array([[1,2],[3,4],[5,6]])              #(3,2) 2개짜리가 3개
a2=np.array([[1,2,3],[4,5,6]])                #(2,3) 3개짜리가 2개
a3=np.array([[[1],[2],[3]],[[4],[5],[6]]])    #(2,3,1) 1개짜리가 3묶음, 그 묶음이 2개
a4=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])            #(2,2,2) 2개짜리가 2묶음, 그 묶음이 2개
a5=np.array([[[1,2,3]],[[4,5,6]]])            #(2,1,3) 3개짜리가 1묶음, 그 묶음이 2개
a6=np.array([1,2,3,4,5,6])                    #(5, ) 
a7=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) #(4,3) 3개짜리가 4묶음


print(a1.shape)
print(a2.shape)
print(a3.shape)
print(a4.shape)
print(a5.shape)
print(a6.shape)
print(a7.shape)

