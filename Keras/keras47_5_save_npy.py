import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

#1. DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255
) #평가는 증폭해서는 안된다 


#이미지 폴더 정의 # D:\_data\image\brain
xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size=(150, 150), #사이즈는 지정된대로 바꿀수있다
    batch_size=200, #배치사이즈 모르면 적당히 넉넉히 잡으면 최대치로 들어간다
    class_mode='binary',
    shuffle=True) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=200, #배치사이즈 모르면 적당히 넉넉히 잡으면 최대치로 들어간다 #짜투리 남는것도 한배치로 돈다
    class_mode='binary') #셔플은 필요없음
    #Found 120 images belonging to 2 classes.

# print(xy_train) #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000015005914F70>
# print(xy_train[0]) #첫번째 배치가 보임  dtype=float32), array([0., 0., 1., 1., 1.], dtype=float32)) 배치사이즈5를 줬기때문에 5개가 나옴
# print(xy_train[31]) #마지막 배치
#print(xy_train[32]) #ValueError: Asked to retrieve element 32, but the Sequence has length 32
#32개 밖에 없으므로 33번째를 호출하면 에러 // 0~5까지 배치사이즈를 나눴기 때문에

#print(xy_train[0][0]) # X 첫번째 배치의 첫X
#print(xy_train[0][1]) # Y
#print(xy_train[0][2]) # IndexError: tuple index out of range

# print(xy_train[0][0].shape) #(5, 100, 100, 3) #디폴트 채널은 3, 컬러다
# print(xy_train[0][1].shape) #(5,)
# print(type(xy_train))       #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>

print(xy_train[0][0].shape, xy_train[0][1].shape)
print(xy_test[0][0].shape, xy_test[0][1].shape)

np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])

