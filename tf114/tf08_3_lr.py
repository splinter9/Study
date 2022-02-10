# 실습
# lr을 수정해서 epochs 100 이하로 줄여라!!
# step = 100 이하, w = 1.9999, b = 0.9999 


import tensorflow as tf
tf.set_random_seed(66)


#.1 DATA
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])



#predict를 위하여 x_test의 placeholder를 생성

x_test = tf.placeholder(tf.float32, shape=[None])



# 가중치와 bias를 초기화

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


#2 학습을 위한 모델 구성

hypothesis = x_train * w + b  # y = wx + b

#3-1 컴파일

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) #mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)


# x_train과 y_train을 이용하여 훈련!


#3-2 훈련

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    # sess.run(train)    
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    if step % 200 == 0:
        # print(step+1, sess.run(loss), sess.run(w),sess.run(b))
        print(step,loss_val,w_val,b_val)




predict = x_test * w + b
print(sess.run(predict, feed_dict={x_test:[8,9]}))
sess.close()


############### 실습과제 예시 #################

# x_data = [6,7,8]
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predict = x_test * w_val
