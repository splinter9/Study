import autokeras as ak
# from tensorflow.keras.datasets import mnist =>
import tensorflow as tf

#1.DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#2.MODEL
model = ak.ImageClassifier(overwrite=True, max_trials=2) #모델 두번돌림

#3.COMPILE
model.fit(x_train, y_train, epochs=5)

#4.EVALUATE
results = model.evaluate(x_test, y_test)
print(results)
print(model.summary())
