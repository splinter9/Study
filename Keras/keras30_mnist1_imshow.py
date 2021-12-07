import numpy as np 
from tensorflow.keras.datasets import mnist #우편번호 손글씨

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0])
print('y_train[0]번째 값 : ', y_train[0])  ## y는 10개 (0,1,2,3,4,5,6,7,8,9)

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')
plt.show()
