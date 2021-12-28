import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy


# 1. 데이터

#이미지 폴더 정의 # D:\_data\image\rps
train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split 

train_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/rps',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical',
    subset='training') # set as training data
#Found 1764 images belonging to 3 classes.

validation_generator = train_datagen.flow_from_directory(
   '../_data/image/rps/rps', # same directory as training data
    target_size=(150,150),
    batch_size=10,
    class_mode='categorical',
    subset='validation') # set as validation data
#Found 756 images belonging to 3 classes.

print(train_generator[0][0].shape)  #
print(validation_generator[0][0].shape) #

test_datagen = ImageDataGenerator(rescale = 1./255)

np.save('../_save_npy/keras48_3_train_x.npy', arr = train_generator[0][0])
np.save('../_save_npy/keras48_3_train_y.npy', arr = train_generator[0][1])
np.save('../_save_npy/keras48_3_test_x.npy', arr = validation_generator[0][0])
np.save('../_save_npy/keras48_3_test_y.npy', arr = validation_generator[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout


model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])


hist = model.fit_generator(train_generator, epochs = 30, steps_per_epoch = 72, 
                    validation_data = validation_generator,
                    validation_steps = 4,)


model.save_weights('./_save/keras48_3_save_weights.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 그래프 그려 보기
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

epochs = range(1, len(loss)+1)
import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'r--', label = 'loss')
plt.plot(epochs, val_loss, 'r:', label = 'val_loss')
plt.plot(epochs, acc, 'b--', label = 'acc')
plt.plot(epochs, val_acc, 'b:', label = 'val_acc')

plt.grid()
plt.legend(['loss','val_loss','acc','val_acc'], loc='upper left')
plt.show


# summarize history for accuracy
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
