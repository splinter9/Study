# y = wx + b
from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(77)


#.1 DATA
# x_train = [1,2,3]
# y_train = [1,2,3]
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])



# w = tf.Variable(1, dtype=tf.float32)
# b = tf.Variable(1, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w)) #[0.06524777]


#.2 MODEL
hypothesis = x_train * w + b   # y = wx + b



#.3-1 COMPILE 
loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) # optimizer='sgd'
# model.copile(loss='mse', optimizer='sgd')



#.3-2 TRAIN
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))

sess.close()  # 세션 종료를 해줘야 메모리가 복귀된다

