# breast_cancer 이진분류 

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
tf.set_random_seed(104)

dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

print(x_data.shape)
print(y_data.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.zeros([30,1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))



train = tf.train.GradientDescentOptimizer(learning_rate=0.000000001).minimize(loss)
 
predict = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([loss, train], feed_dict = {x:x_data, y:y_data})

        if step % 2000 == 0:
            print(step, cost_val) 


    h , c, a = sess.run([hypothesis,predict,accuracy],feed_dict = {x:x_data,y:y_data})
    print("예측값 : \n",h)#,"\n원래값 : \n", c, "\nacc : ", a)

    acc = accuracy_score(y_data, sess.run(predict, feed_dict={x:x_data}))
    print("accuracy_score : ", acc)


# accuracy_score :  0.37258347978910367