import tensorflow as tf


tf.set_random_seed(104)

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]


x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

# 다층 레이어 구성

w1 = tf.compat.v1.Variable(tf.random_normal([2,64]), name='weight1') #  [입력, 출력]
b1 = tf.compat.v1.Variable(tf.random_normal([64]), name = 'bias1') # weight의 출력과 같다
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([64,32]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([32]), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)


w3 = tf.compat.v1.Variable(tf.random_normal([32,1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3) # 최종 아웃풋


# 위에 모델을 Dense로 구성했다면 아래와 같다
# model.add(Dense(64, input_dim = 2,activation = 'sigmoid'))
# model.add(Dense(32, activation = 'sigmoid'))
# model.add(Dense(1, activation = 'sigmoid'))


# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # loss
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
predict = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5601):
        loss_val, _ = sess.run([loss, train], feed_dict = {x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, loss_val) 


    h , c, a = sess.run([hypothesis,predict,accuracy],feed_dict = {x:x_data,y:y_data})
    print("예측값 : \n",h,"\n원래값 : \n", c, "\n acc : ", a)