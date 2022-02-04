import autokeras as ak
# from tensorflow.keras.datasets import mnist =>
import tensorflow as tf

#1.DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

#2.MODEL
model = ak.ImageClassifier(overwrite=True, max_trials=5)

#3.COMPILE
model.fit(x_train, y_train, epochs=10)
results = model.evaluate(x_test, y_test)
print(results)




'''
[2.6626784801483154, 0.33889999985694885]

'''