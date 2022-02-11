from turtle import update
from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(77)

x_train_data = [1, 2, 3]
y_train_data = [1, 2, 3]
x_test_data = [4, 5, 6]
y_test_data = [4, 5, 6]


x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y))

lr = 0.1
gradient = tf.reduce_mean((w * x -y)* x)
descent = w - lr * gradient
update = w.assign(descent) # w = w - lr * gradient

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history=[]
loss_history=[]

for step in range(21):
    # sess.run(update, feed_dict={x:x_train, y:y_train})
    # print(step, '\t', sess.run(loss, feed_dict={x:x_train, y:y_train}), sess.run(w))
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train_data, y:y_train_data})
    print(step,'\t', loss_v,'\t', w_v)
    
    # w_history.append(w_v)
    # loss_history.append(loss_v)    

from sklearn.metrics import r2_score, mean_absolute_error

y_predict = x_test * w_v
y_predict_data = sess.run(y_predict, feed_dict={x_test:x_test_data})
print('y_predict_data:', y_predict_data)

sess.close()

r2 = r2_score(y_test_data, y_predict_data)
print('R2 score:', r2)

mae = mean_absolute_error(y_test_data, y_predict_data)
print('mae:', mae)

# y_predict_data: [3.999996  4.9999948 5.999994 ]
# R2 score: 0.9999999999588169
# mae: 5.1657358805338544e-06


# print("============== w history ======================")
# print(w_history)
# print("============== loss history ===================")
# print(loss_history)


# plt.plot(w_history, loss_history)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.show()




