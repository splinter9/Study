import numpy as np
import tensorflow as tf

tf.set_random_seed(66)

x_data =[[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,6,7]]

y_data =[[0,0,1],
         [0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

x_predict = [[]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])  # 인풋 4
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])  # 인풋 3
# 행열 연산은 앞의 열과 뒤의 행의 shape이 맞아야 함 w의 행은 x의 열의 갯수 와 동일한 shape이여야 함  


w = tf.compat.v1.Variable(tf.random.normal([4,1]), name='weight')    # y = x * w  ;  (5, 1) = (5, 3) * (? * ?)  => (3, 1)    
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name='bias')        # bias는 덧셈이므로 shape변화 없음


#2. 모델
# hypothesis = x * w + b
# hypothesis = tf.matmul(x, w) + b

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(1, activation='sigmoid'))


#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))                         # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binarycrossentropy
loss = -tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))     # categoricalcrossentropy 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, loss_val)




#4. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

feed_dicts= {x:x_data, y:y_data}

for epoch in range(70001):
    sess.run(optimizer, feed_dict=feed_dicts)
    if epoch % 2000 == 0:
        print(epoch, sess.run(loss, feed_dict=feed_dicts))
    

#5. 평가
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 확률값이 0.5 이상이면 1, 아니면 0

#6. 예측

#7. 테스트

#8. 성능평가

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32)) # 예측값과 실제값이 같으면 1, 아니면 0

pred, acc = sess.run([y_predict, accuracy], feed_dict=feed_dicts)

print("예측결과 :", pred, "\n", "ACC :", acc)