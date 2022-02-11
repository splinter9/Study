from sklearn.datasets import load_wine
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


tf.set_random_seed(66)

dataset = load_wine()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

print(x_data.shape) # (178, 13)
print(y_data.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=77)


x = tf.compat.v1.placeholder('float',shape=[None,13])
y = tf.compat.v1.placeholder('float',shape=[None,3])

w = tf.compat.v1.Variable(tf.zeros([13,3],name = 'weight'))
b = tf.compat.v1.Variable(tf.zeros([1,3]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)


    # y_acc_test = sess.run(tf.argmax(y_test, 1)) # predict와 맞춰줘야지 accuracy_score이 작동한다.
    # predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    # acc = accuracy_score(y_acc_test, predict)
    # print("accuracy_score : ", acc)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)
    


'''
0 nan
200 nan
400 nan
600 nan
800 nan
1000 nan
1200 nan
1400 nan
1600 nan
1800 nan
2000 nan

acc :  0.44444445
'''