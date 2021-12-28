import numpy as np

# np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')

print(x_train)
print(x_train.shape)

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs = 30)
acc = hist.history['acc']
#val_acc = hist.history['val_loss']
loss = hist.history['loss']
#val_loss = hist.history['val_loss']

print('loss:', loss[-1])
#print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
#print('val_acc:',val_acc [-1])
