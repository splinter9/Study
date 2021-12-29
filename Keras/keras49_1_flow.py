from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    #vertical_flip=True,             #상하반전
    width_shift_range=0.1,
    height_shift_range=0.1,
    #rotation_range=5,
    zoom_range=0.1,
    #shear_range=0.7,
    fill_mode='nearest'
)

print(x_train[0].shape)                    #(28, 28)
print(x_train[0].reshape(28*28).shape)     #(784,)

augment_size = 100 # 증폭??
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1, 28, 28, 1), #x_train의 0번째 이미지를 28*28 로 쭉늘린다는 뜻, 2차원이 안들어감 1차원으로 변환
    np.zeros(augment_size), 
    batch_size=augment_size,
    shuffle=False
).next() #.next()를 빼면 실행안됨 

print(type(x_data)) #<class 'tuple'>
print(x_data[0].shape, x_data[1].shape) #(100, 28, 28, 1) (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
















'''
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

test_datagen = ImageDataGenerator(
    rescale=1./255
) #평가는 증폭해서는 안된다 


#이미지 폴더 정의 
# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size=(100, 100), #사이즈는 지정된대로 바꿀수있다
    batch_size=5, 
    class_mode='binary',
    shuffle=True, seed=42, color_mode="grayscale"
    ) #Found 160 images belonging to 2 classes.



xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=5, #짜투리 남는것도 한배치로 돈다
    class_mode='binary') #셔플은 필요없음
    #Found 120 images belonging to 2 classes.




print(xy_train) #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000015005914F70>
print(xy_train[0]) #첫번째 배치가 보임  dtype=float32), array([0., 0., 1., 1., 1.], dtype=float32)) 배치사이즈5를 줬기때문에 5개가 나옴
print(xy_train[31]) #마지막 배치
#print(xy_train[32]) #ValueError: Asked to retrieve element 32, but the Sequence has length 32
#32개 밖에 없으므로 33번째를 호출하면 에러 // 0~5까지 배치사이즈를 나눴기 때문에

#print(xy_train[0][0]) #X 첫번째 배치의 첫X
#print(xy_train[0][1]) #Y
#print(xy_train[0][2]) #IndexError: tuple index out of range

print(xy_train[0][0].shape) #(5, 100, 100, 3) #디폴트 채널은 3, 컬러다
print(xy_train[0][1].shape) #(5,)
print(type(xy_train))       #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
'''
