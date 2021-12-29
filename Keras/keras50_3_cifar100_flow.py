#훈련데이터 10만개로 증폭
#완료후 기존 모델과 비교
#save_dir도 _temp에 넣고
#증폭데이터는 _temp에 저장후 훈련 끝나고 삭제
'''
from numpy.random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
test_datagen = ImageDataGenerator(
    rescale=1./255)


# print(x_train[0].shape)                    #(32, 32, 3)
# print(x_train[0].reshape(32*32*3).shape)   #(3072,)
# print(x_train.shape)                    #(50000, 32, 32, 3)


augment_size = 50000
randint = np.random.randint(x_train.shape[0], size=augment_size)

# print(x_train.shape[0]) #
# print(randint) #
# print(np.min(randint), np.max(randint)) #

x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()
# print(x_augmented.shape) #(100000, 32, 32, 3)
# print(y_augmented.shape) #(100000, 1)

# x_augmented = x_augmented.reshape(x_augmented.shape[0],
#                                   x_augmented.shape[1],
#                                   x_augmented.shape[2],3)
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(x_test.shape[0],32,32,3)



xy_train = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size, shuffle=False,)#.next()[0]
                                 #save_to_dir='../_temp').next()[0]
                                 
xy_test = test_datagen.flow(x_test, y_test,
                                 batch_size=augment_size, shuffle=False,)#.next()[0]
                                 #save_to_dir='../_temp').next()[0]
#print(x_augmented) #
# print(x_augmented.shape) #(100000, 32, 32, 3)

x_train = np.concatenate((x_train, xy_train)) #(50000, 32, 32, 3)
y_train = np.concatenate((y_train, xy_train)) #(50000, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# xy_train = to_categorical(xy_train)

# x_train, x_test_, y_train, y_test_ = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

model = Sequential()
model.add(Conv2D(300, (2,2), input_shape=(32,32,3)))
model.add(Conv2D(200, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(52))
model.add(Dense(40))
model.add(Dense(28))
model.add(Dense(16))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#es=EarlyStopping(monitor='val_loss', patience=1, mode='auto', verbose=1, restore_best_weights=True)
model.fit(xy_train, epochs=10, batch_size=32, validation_split=0.2,) #callbacks=[es])
loss = model.evaluate(xy_test)

print(loss)
# y_predict=model.predict(xy_test)
# y_predict=np.argmax(y_predict,axis=1)
# y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_test, y_predict)
# print('accuracy score:' , acc)
# print('loss : ', loss)
'''


from re import X
import numpy as np
import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size=augment_size) # randint - 랜덤한 정수값을 뽑는다
print(x_train.shape[0]) # 50000

print(randidx) # [32882 21036 43516 ... 48177 49866 51437]
print(np.min(randidx), np.max(randidx)) # 0 ~ 59996

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) # (50000, 28, 28)
print(y_augmented.shape) # (50000,)

# x_augmented = x_augmented.reshape(x_augmented.shape[0], 
#                                   x_augmented.shape[1],
#                                   x_augmented.shape[2], 3)
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)


x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir = '../_temp/'
                                 ).next()[0]
print(x_augmented)
print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train)
print(x_train.shape) # augmented 와 합쳐진 x_train mnist 값 (100000, 32, 32, 3)
print(y_test.shape) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_test.shape) # (10000, 32, 32, 3)
print(y_train.shape)# (100000, 100)
print(y_test.shape)# (10000, 100)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), strides = 1,
                 padding='valid', input_shape = (32, 32, 3))) #padding = same, valid
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation = 'relu')) # 7,7,5
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1200, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(640, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(320, activation = 'relu'))
model.add(Dense(160, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'fashion_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=5, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=15, batch_size = 32, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/mnist_1227_1730_0016-0.3595.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])


y_pred = model.predict(x_test)
result_recover = np.argmax(y_pred, axis=1)
result_testrecover = np.argmax(y_test, axis=1)
# print(result_recover)
# print(result_testrecover)
result = accuracy_score(result_recover,result_testrecover)

print('acc_score : ', result)