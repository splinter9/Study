from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(66)
     #    [1ST  2ND  3RD  4TH  5TH]    
x1_data = [73., 93., 89., 96., 73.]      # 국어
x2_data = [89., 88., 91., 98., 66.]      # 영어
x3_data = [75., 93., 90., 100., 70.]     # 수학 .을 찍은 이유는 float
y_data = [152., 185., 180., 196., 142.]  # 환산점수 
                                         # 5행3열 데이터
# x는 (5,3)  y는 (5,1) 또는 (5,)
# y = x1 * w1 + x2 * w2 + x3 * w3

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([w1, w2, w3]))

#2. 모델

hypothesis = x1*w1 + x2*w2 + x3*w3 + b


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(1001):
    _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print(epochs, '\t', loss_val, '\t', w_val1, '\t', w_val2, '\t', w_val3)
    
   
#4. 예측
predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)



'''
예측 :  [152.0936  185.09546 179.71109 196.06471 142.05089]
r2스코어 :  0.9999486125778573
mae :  0.11871337890625
'''