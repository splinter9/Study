import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy


# 1. 데이터

# np.save('../_save_npy/keras48_4_train_x.npy', arr = train_generator[0][0])
# np.save('../_save_npy/keras48_4_train_y.npy', arr = train_generator[0][1])
# np.save('../_save_npy/keras48_4_test_x.npy', arr = validation_generator[0][0])
# np.save('../_save_npy/keras48_4_test_y.npy', arr = validation_generator[0][1])

x_train = np.load('../_save_npy/keras48_4_train_x.npy')
x_test = np.load('../_save_npy/keras48_4_test_x.npy')
y_train = np.load('../_save_npy/keras48_4_train_y.npy')
y_test = np.load('../_save_npy/keras48_4_test_y.npy')

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
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])


hist = model.fit(x_train, y_train, epochs=30, batch_size=1, validation_split=0.2)


model.save_weights('./_save/keras48_4_save_weights.h5')

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
plt.legend()
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
