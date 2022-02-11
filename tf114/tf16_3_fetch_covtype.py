from sklearn.datasets import fetch_covtype
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


tf.set_random_seed(104)

dataset = fetch_covtype()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=104)


x = tf.compat.v1.placeholder('float',shape=[None,54])
y = tf.compat.v1.placeholder('float',shape=[None,7])

w = tf.compat.v1.Variable(tf.zeros([54,7]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,7]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)
    
    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)
    


'''
0 1.9459207
200 1.1005179
400 1.0714806
600 1.0470115
800 1.0247437
1000 1.0045651
1200 0.9863758
1400 0.9699947
1600 0.95521206
1800 0.94182223
2000 0.92964077

acc :  0.6346842
accuracy_score :  0.6346842298512942
'''