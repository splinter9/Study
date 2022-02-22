from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time
import numpy as np
from tensorflow.keras.datasets import cifar100

model01 = VGG16(include_top=False, input_shape=(32,32,3))
model02 = VGG19(include_top=False, input_shape=(32,32,3))
model03 = ResNet50(include_top=False, input_shape=(32,32,3))
model04 = ResNet50V2(include_top=False, input_shape=(32,32,3))
model05 = ResNet101(include_top=False, input_shape=(32,32,3))
model06 = ResNet101V2(include_top=False, input_shape=(32,32,3))
model07 = ResNet152(include_top=False, input_shape=(32,32,3))
model08 = ResNet152V2(include_top=False, input_shape=(32,32,3))
model09 = DenseNet121(include_top=False, input_shape=(32,32,3))
model10 = DenseNet169(include_top=False, input_shape=(32,32,3))
model11 = DenseNet201(include_top=False, input_shape=(32,32,3))
model12 = InceptionV3(include_top=False, input_shape=(32,32,3))
model13 = InceptionResNetV2(include_top=False, input_shape=(32,32,3))
model14 = MobileNet(include_top=False, input_shape=(32,32,3))
model15 = MobileNetV2(include_top=False, input_shape=(32,32,3))
model16 = NASNetLarge(include_top=False, input_shape=(32,32,3))
model17 = NASNetMobile(include_top=False, input_shape=(32,32,3))
model18 = EfficientNetB0(include_top=False, input_shape=(32,32,3))
model19 = EfficientNetB1(include_top=False, input_shape=(32,32,3))
model20 = EfficientNetB7(include_top=False, input_shape=(32,32,3))
model21 = Xception(include_top=False, input_shape=(32,32,3))
model22 = MobileNetV3Small(include_top=False, input_shape=(32,32,3))
model23 = MobileNetV3Large(include_top=False, input_shape=(32,32,3))

model_list = [model01,model02,model03,model04,model05, model06,model07,model08,model09,model10,
              model11,model12,model13,model14,model15,model16,model17,model18,model19,model20,
              model21,model22,model23]


(x_train, y_train),(x_test,y_test) = cifar100.load_data()
out_node = len(np.unique(y_train))

x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

for model in model_list:
    model_name = model.name    
    print("모델명 : " + model_name)
    try:
        model.trainable = False     
        models = Sequential()
        models.add(model)
        # model.add(Flatten())
        models.add(GlobalAveragePooling2D())
        models.add(Dense(100))
        models.add(Dense(10, activation='softmax'))

        #3. 컴파일, 훈련
        opt="adam"
        models.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
        ########################################################################
        # model.compile(loss = 'mse', optimizer = 'adam')
        start = time.time()
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        import datetime
        epoch = 10
        patience_num = 50
        date = datetime.datetime.now()
        datetime = date.strftime("%m%d_%H%M")
        es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
        #mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
        models.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es], batch_size = 1)
        end = time.time() - start
        ########################################################################
        #4 평가예측
        loss = models.evaluate(x_test,y_test)
        y_predict = models.predict(x_test)
        print('시간 : ', round(end,2) ,'초')
        print("loss : ",round(loss[0],4))
        print("accuracy : ",round(loss[1],4))
    except:
        print(print("모델 오류 : " + model_name))
        
