### 내얼굴 넣은 버전###
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image

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
) #Found 8005 images belonging to 2 classes.

test_datagen = ImageDataGenerator(
    rescale=1./255
) #Found 2023 images belonging to 2 classes.


#이미지 폴더 정의 # D:\_data\image\cat_dog
xy_train = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set/',
    target_size=(250, 250), #사이즈는 지정된대로 바꿀수있다
    batch_size=3,
    class_mode='binary',
    shuffle=True)

xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/',
    target_size=(250, 250),
    batch_size=3, #짜투리 남는것도 한배치로 돈다
    class_mode='binary') #셔플은 필요없음

##개와 고양이는 애초 트레인과 테스트가 분류되어 있으므로 train_generator = train_datagen 안써도 된다


print(xy_train) #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000015005914F70>
print(xy_train[0]) #첫번째 배치가 보임  dtype=float32), array([0., 0., 1., 1., 1.], dtype=float32)) 배치사이즈5를 줬기때문에 5개가 나옴
print(xy_train[31]) #마지막 배치
#print(xy_train[32]) #ValueError: Asked to retrieve element 32, but the Sequence has length 32
#32개 밖에 없으므로 33번째를 호출하면 에러 // 0~5까지 배치사이즈를 나눴기 때문에

#print(xy_train[0][0]) #X 첫번째 배치의 첫X
#print(xy_train[0][1]) #Y
#print(xy_train[0][2]) #IndexError: tuple index out of range

print(xy_train[0][0].shape) #(1, 250, 250, 3)
print(xy_train[0][1].shape) #(1,)
print(type(xy_train))       #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>

np.save('../_save_npy/keras48_1_train_x.npy', arr=xy_train[0][0])
np.save('../_save_npy/keras48_1_train_y.npy', arr=xy_train[0][1])
np.save('../_save_npy/keras48_1_test_x.npy', arr=xy_test[0][0])
np.save('../_save_npy/keras48_1_test_y.npy', arr=xy_test[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(250, 250, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

# model.fit(xy_train[0][0], xy_train[0][1])    # x 와 y   validation_steps = 4, 뜻 알아내기
hist = model.fit_generator(xy_train, epochs = 30, steps_per_epoch = 32, # steps_per_epoch : 에포당 스텝을 몇 번할 것인가?? = 전체 데이터 나누기 배치
                    validation_data = xy_test,
                    validation_steps = 4,)

model.save_weights('./_save/keras48_1_save_weights.h5')
model.save("./_save/keras48_1_save_model.h5")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 그래프 그려 보기
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])


import matplotlib.pyplot as plt
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'r--', label = 'loss')
plt.plot(epochs, val_loss, 'r:', label = 'val_loss')
plt.plot(epochs, acc, 'b--', label = 'acc')
plt.plot(epochs, val_acc, 'b:', label = 'val_acc')

plt.grid()
plt.legend()
plt.show
        
        
# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/_predict/cat_dog/'
sample_image = sample_directory + "han1.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

print("-- Evaluate --")
scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = keras_image.load_img(str(sample_image), target_size=(250, 250))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
xy_test.reset()
print(xy_test.class_indices)
# {'cat': 0, 'dog': 1}
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 cat 입니다")
elif(classes[0][0]>0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 dog 입니다")
else:
    print("ERROR")


