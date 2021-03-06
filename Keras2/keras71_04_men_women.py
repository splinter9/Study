import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.image import image_data_generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode= 'nearest')   
 
test_datagen = ImageDataGenerator(rescale=1./255) 

xy_train = train_datagen.flow_from_directory(
    '../_data/image/menwomen/men_women/', 
    target_size = (100,100), 
    batch_size=3,
    class_mode = 'binary',
    shuffle= True) 
#Found 3309 images belonging to 1 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/menwomen/men_women/',
    target_size = (100,100),
    batch_size = 3,
    class_mode = 'binary') 
#Found 3309 images belonging to 1 classes.

# np.save('../_data/_save_npy/keras48_4_train_x.npy', arr = xy_train[0][0])  
# np.save('../_data/_save_npy/keras48_4_train_y.npy', arr = xy_train[0][1])
# np.save('../_data/_save_npy/keras48_4_test_x.npy', arr = xy_test[0][0])
# np.save('../_data/_save_npy/keras48_4_test_y.npy', arr = xy_test[0][1])

x_train = np.load('../_data/_save_npy/keras48_4_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_4_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_4_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_4_test_y.npy')

#2. 모델링
from tensorflow.keras.applications import DenseNet121, VGG19
import time
vgg19 = VGG19(weights = 'imagenet', include_top= False, input_shape= (100, 100, 3))

#densenet121.trainable = False  # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg19)
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)  

model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, mode = 'auto', verbose = 1, factor = 0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))

