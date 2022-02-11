from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


tf.set_random_seed(104)

dataset = load_iris()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=104)


x = tf.compat.v1.placeholder('float',shape=[None,4])
y = tf.compat.v1.placeholder('float',shape=[None,3])

w = tf.compat.v1.Variable(tf.random_normal([4,3],name = 'weight'))
b = tf.compat.v1.Variable(tf.random_normal([1,3]), name = 'bias')

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

0 18.194534
200 0.5615492
400 0.4951144
600 0.44657883
800 0.4084558
1000 0.37743747
1200 0.35165855
1400 0.3299065
1600 0.31132808
1800 0.29529542
2000 0.281333

acc :  0.95555556

'''