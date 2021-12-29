#훈련데이터 10만개로 증폭
#완료후 기존 모델과 비교
#save_dir도 _temp에 넣고
#증폭데이터는 _temp에 저장후 훈련 끝나고 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
import matplotlib.pyplot as plt

# 1. 데이터

# 클래스에 대한 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest'          
    )                      

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/brain/train',
    target_size = (150,150),                         
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size = (150,150),
    batch_size = 5, 
    class_mode = 'binary',
)

print(xy_train[0][0].shape, xy_train[0][1].shape) 


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (150,150,1)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

# model.fit(xy_train[0][0], xy_train[0][1])    # x 와 y   validation_steps = 4, 뜻 알아내기
hist = model.fit_generator(xy_train, epochs = 30, steps_per_epoch = 32, # steps_per_epoch : 에포당 스텝을 몇 번할 것인가?? = 전체 데이터 나누기 배치
                    validation_data = xy_test,
                    validation_steps = 4,)  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']