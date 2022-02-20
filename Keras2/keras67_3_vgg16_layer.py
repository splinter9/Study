import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16()
vgg16 = VGG16()
vgg16.summary()

model = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

vgg16.trainable = False  # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.summary()

print(len(model.weights))             # 30
print(len(model.trainable_weights))   # 30


################## 2번에서 아래만 추가 ##################

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type','Layer Name', 'Layer Trainable'])
print(results)
