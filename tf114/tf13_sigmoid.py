import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]   # (6, 2)
y_data = [[0],[0],[0],[1],[1],[1]]               # (6, 1)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])  # 인풋 2
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])  # 인풋 1
# 행열 연산은 앞의 열과 뒤의 행의 shape이 맞아야 함 w의 행은 x의 열의 갯수 와 동일한 shape이여야 함  


w = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight')    # y = x * w  ;  (5, 1) = (5, 3) * (? * ?)  => (3, 1)    
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')        # bias는 덧셈이므로 shape변화 없음


#2. 모델
# hypothesis = x * w + b
# hypothesis = tf.matmul(x, w) + b

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# model.add(Dense(1, activation='sigmoid'))


#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binarycrossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004)
train = optimizer.minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val ,_= sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
       print(epochs, loss_val, '\t', hy_val)
    
   
#4. 예측

y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32) # .cast 함수는 Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(y,y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print('======================================')
print('예측값: \n', hy_val)
print('예측결과:', pred)
print('Accuracy:', acc)


sess.close()


'''
======================================
예측값:
 [[0.2703899 ]
 [0.29318857]
 [0.7168443 ]
 [0.6168474 ]
 [0.7602875 ]
 [0.9166514 ]]
예측결과: [[0.]
 [0.]
 [1.]
 [1.]
 [1.]
 [1.]]
Accuracy: 0.8333333
'''