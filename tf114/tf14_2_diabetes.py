from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import r2_score

dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)


x = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([10,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random_normal([1]),name = 'bias')

# hypothesis = x*w+b
hypothesis = tf.matmul(x, w) + b # matmul = 곱하기

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=(1e-6))
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(20001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 2000 == 0:
        print(step, 'loss : ', loss_val,"\n 예측값 : \n", hy_val)
    r2 = r2_score(y_data, hy_val)
    print('R2: ', r2)

sess.close()

# R2:  -3.8753074305294204
