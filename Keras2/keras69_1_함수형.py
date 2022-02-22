from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16, Xception


input = Input(shape=(100, 100, 3))
vgg16 = VGG16(include_top=False)(input)
hidden1 = Dense(100)(vgg16)
hidden2 = Dense(50, activation='relu')(hidden1)
hidden3 = Dense(30, activation='relu')(hidden2)
output1 = Dense(10)(hidden3)

model = Model(inputs=input, outputs=output1)

model.summary()


'''

#3. 컴파일, 훈련

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))
'''