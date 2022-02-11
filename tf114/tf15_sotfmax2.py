import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
# (8,4)
y_data = [[0,0,1],      # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],      # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],      # 0
          [1,0,0]]
# (8,3)

#2. 모델 구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random.normal([1, 3]), name = 'bias')  # 컬럼에 맞춰서 늘려준다.

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) 

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.08).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
    results = sess.run(hypothesis, feed_dict = {x : x_data})
    print(results, sess.run(tf.math.argmax(results, 1)))
    accuracy =tf.reduce_mean(tf.cast(tf.equal(y_data, results), dtype = tf.float32))
    print( 'accuracy :', sess.run(accuracy))
    
'''
0 12.071602
200 0.50911826
400 0.427224
600 0.38839573
800 0.36146185
1000 0.33993608
1200 0.32162824
1400 0.3055375
1600 0.291113
1800 0.27801389
2000 0.26601082
[[1.0176286e-05 2.1022307e-03 9.9788755e-01]
 [1.7990460e-05 1.3592325e-01 8.6405873e-01]
 [4.4032356e-07 1.4989664e-01 8.5010290e-01]
 [9.2214883e-08 8.4009683e-01 1.5990306e-01]
 [4.7072029e-01 5.1161885e-01 1.7660843e-02]
 [2.3066506e-01 7.6925874e-01 7.6168559e-05]
 [6.2839448e-01 3.7145561e-01 1.4984357e-04]
 [7.8217959e-01 2.1775798e-01 6.2422791e-05]] [2 2 2 1 1 1 0 0]
accuracy : 0.6666667
'''