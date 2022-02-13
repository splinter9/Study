import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
(x_train, y_train), (x_test,  y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_train)
y_train= ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

x = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,10)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,10)

# 레이어를 쌓아준다.

w1 = tf.compat.v1.Variable(tf.zeros([784,128]), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([1,128]), name = 'bias')
layer1 = tf.matmul(x,w1) + b1

w2 = tf.compat.v1.Variable(tf.random_normal([128,32]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([1,32]), name = 'bias2')
layer2 = tf.matmul(layer1,w2) + b2


w3 = tf.compat.v1.Variable(tf.random_normal([32,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([1,10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3) # 최종 아웃풋


# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)

    # a = sess.run(hypothesis, feed_dict={x:x_test})
    # print(a,'\n' ,sess.run(tf.argmax(a, 1)))
        y_acc_test = sess.run(tf.argmax(y_test, 1))
        predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
        acc = accuracy_score(y_acc_test, predict)
        print(step,"accuracy_score : ", acc)

sess.close()
