from numpy.random import randint
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.core import Activation

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

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size) # randint - 랜덤한 정수값을 뽑는다

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy() 

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


xy_train = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augment_size),
                                 batch_size=32, shuffle=False
                                 )#.next()[0]

# print(xy_train)
# print(xy_train[0].shape, xy_train[1].shape) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(len(xy_train))
model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train))


from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2값은: ', r2)

loss= model.evaluate(x_test, y_test)
print('loss: ', loss)


################# 은탁버전 ######################
'''
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size) # randint - 랜덤한 정수값을 뽑는다
print(x_train.shape[0]) # 60000

print(randidx) # [32882 21036 43516 ... 48177 49866 51437]
print(np.min(randidx), np.max(randidx)) # 0 ~ 59996

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) # (40000, 28, 28)
print(y_augmented.shape) # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


xy_train = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augment_size),
                                 batch_size=32, shuffle=False
                                 )#.next()[0]

# print(xy_train)
# print(xy_train[0].shape, xy_train[1].shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape = (28,28,1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(len(xy_train))
model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train))
'''