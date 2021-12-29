#훈련데이터 10만개로 증폭
#완료후 기존 모델과 비교
#save_dir도 _temp에 넣고
#증폭데이터는 _temp에 저장후 훈련 끝나고 삭제


from numpy.random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
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
test_datagen = ImageDataGenerator(
    rescale = 1./255
)

print(x_train[0].shape)                    #(28,28)
print(x_train[0].reshape(28*28).shape)     #(784,)

augment_size = 100000
randint = np.random.randint(x_train.shape[0], size=augment_size)

print(x_train.shape[0]) #60000
print(randint) #[24985 33694 34164 ... 30917 13026 41366]
print(np.min(randint), np.max(randint)) #0 59997

x_augmented = x_train[randint].copy()
y_augmented = y_train[randint].copy()
print(x_augmented.shape) #(100000, 28, 28)
print(y_augmented.shape) #(100000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)
x_train = x_train.reshape(60000, 28, 28 ,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='../_temp').next()[0]

x_train=np.concatenate((x_train, x_augmented)) #(40000, 28, 28)
y_train = np.concatenate((y_train, y_augmented)) #(40000,)
#print(x_augmented)
print(x_augmented.shape) #(100000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#x_train, x_test_, y_train, y_test_ = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

model=Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))     
model.add(Conv2D(10,(3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))                     
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#es=EarlyStopping(monitor='val_loss', patience=1, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2,) #callbacks=[es])


loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_predict)
print('accuracy score:' , acc)
print('loss : ', loss)


