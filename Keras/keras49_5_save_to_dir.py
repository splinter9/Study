from numpy.random import randint
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

print(x_train[0].shape)                    #(28, 28)
print(x_train[0].reshape(28*28).shape)     #(784,)

augment_size = 10
randidx = np.random.randint(x_train.shape[0], size=augment_size) 
#정수값 60000개 중에 40000개를 랜덤하게 정수값을 생성한다
#60000장 + 40000장 = 10만장을 훈련시킴
print(x_train.shape[0])  #60000
print(randidx)           #[ 5950 32287 28317 ... 35359 13205 52415]
print(np.min(randidx), np.max(randidx)) #0 59998

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy() 
print(x_augmented.shape)   #(40000, 28, 28)
print(y_augmented.shape)   #(40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

import time
start_time = time.time()

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='../_temp'  #폴더안에 증폭된 이미지가 생성된다.
                                 ).next()[0]
end_time = time.time() - start_time
print('걸린시간: ', round(end_time, 3),'초')

# print(x_augmented)
# print(x_augmented.shape)


# x_train = np.concatenate((x_train, x_augmented)) #괄호가 두개인 이유는?
# y_train = np.concatenate((y_train, y_augmented)) #괄호가 두개인 이유는?

# print(x_train)
# print(x_train.shape) #(100000, 28, 28, 1)

