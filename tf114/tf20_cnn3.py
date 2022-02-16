import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])


#2. 모델 구성

w1 = tf.get_variable('w1', shape = [2,2,1,64]) # [커널 사이즈 = (2,2,1), output]
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME')  #  shape 맞춰주기 위해 허수를 채워줬다. >> 앞 뒤 두 개의 1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# model.add(Conv2d(filters = 64, kernel_size = (2,2), strides = (1,1), padding = 'valid',
#                           input_shape = (28, 28, 1)))
# 커널 사이즈는 가중치였다.

print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 28, 28, 64), dtype=float32)  # 커널사이즈 SAME
print(L1_maxpool)


# Layer2
w2 = tf.get_variable('w2', shape=[3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2_maxpool)


# Layer3
w3 = tf.get_variable('w3', shape=[3, 3, 128, 64])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1,1,1,1], padding = 'SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3_maxpool)


# Layer4
w4 = tf.get_variable('w4', shape=[3, 3, 128, 64],
                     initailizer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides = [1,1,1,1], padding = 'SAME')
L4 = tf.nn.elu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4_maxpool)


# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*32])
print('플래튼: ', L_flat)


# Layer5
w5 = tf.compat.v1.get_variable('w5', shape = [L4.shape[1]*L4.shape[2]*L4.shape[3], 8],initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.compat.v1.Variable(tf.random.normal([8]), name='b5')
L5 = tf.nn.relu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, rate=0.2)
print(L5)


# Layer6
w6 = tf.compat.v1.get_variable('w6', shape = [8, 4],initializer=tf.contrib.layers.variance_scaling_initializer())
b6 = tf.compat.v1.Variable(tf.random.normal([4]), name='b6')
L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)
L6 = tf.nn.dropout(L6, rate=0.2)
print(L6)


# hypothesis
w7 = tf.compat.v1.get_variable('w7', shape = [4, 10],initializer=tf.contrib.layers.variance_scaling_initializer())
b7 = tf.compat.v1.Variable(tf.random.normal([10]), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print(hypothesis)


learning_rate = 0.01
epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)


loss =  tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) #categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# 훈련
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(epochs):
    avg_cost = 0

    for i in range(total_batch):    
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('epoch : ','%04d' %(epoch+1),'cost = {:.9f}'.format(avg_cost))


prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print('acc : ', sess.run(accuracy,feed_dict = {x:x_test,y:y_test}))