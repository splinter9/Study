# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

# 위 값들을 이용하여 predict하라
# x_test라는 placeholder 생성

import tensorflow as tf
tf.set_random_seed(66)



#1 

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


# x_train과 y_train을 이용하여 훈련!


#3-2 훈련

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    # sess.run(train)    
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 100 == 0:
        # print(step+1, sess.run(loss), sess.run(w),sess.run(b))
        print(step,loss_val,w_val,b_val)


#4. 예측
predict = x_test * w_val + b_val      # predict = model.predict

print("[4] 예측 : " , sess.run(predict, feed_dict={x_test:[4]}))
print("[5,6] 예측 : " , sess.run(predict, feed_dict={x_test:[5,6]}))
print("[6,7,8] 예측 : " , sess.run(predict, feed_dict={x_test:[6,7,8]}))

sess.close()


