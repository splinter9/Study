from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score

datasets = load_boston()

x_data = datasets.data
y_data = datasets.target.reshape(-1,1)


print(x_data.shape) # (506, 13)
print(y_data.shape) # (506,)

x = tf.compat.v1.placeholder(tf.float32, shape = [None,13])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])


#  다층레이어
#  모델구성
#   model = Sequential()
#1  model.add(Conv1D(200, 5, input_shape=(13,1))) ##행은 넣지않는다
#2  model.add(Dense(150, activation='relu'))
#3  model.add(Dense(180, activation='relu'))
#4  model.add(Dense(80, activation='linear'))
#5  model.add(Dense(50, activation='relu'))
#6  model.add(Dense(30, activation='relu'))
#7  model.add(Dense(20, activation='relu'))
#8  model.add(Dense(10, activation='linear'))
#9  model.add(Dense(1))


w1 = tf.compat.v1.Variable(tf.random_normal([13,200]), name='weight1') #  [입력, 출력]
b1 = tf.compat.v1.Variable(tf.random_normal([200]), name = 'bias1') # weight의 출력과 같다
layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([200,150]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([150]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random_normal([150,180]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([180]), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([180,80]), name='weight4') 
b4 = tf.compat.v1.Variable(tf.random_normal([80]), name = 'bias4') 
layer4 = tf.nn.relu(tf.matmul(layer3,w4) + b4)

w5 = tf.compat.v1.Variable(tf.random_normal([80,50]), name='weight5') 
b5 = tf.compat.v1.Variable(tf.random_normal([50]), name = 'bias5') 
layer5 = tf.nn.relu(tf.matmul(layer4,w5) + b5)

w6 = tf.compat.v1.Variable(tf.random_normal([50,30]), name='weight6') 
b6 = tf.compat.v1.Variable(tf.random_normal([30]), name = 'bias6') 
layer6 = tf.nn.relu(tf.matmul(layer5,w6) + b6)

w7 = tf.compat.v1.Variable(tf.random_normal([30,20]), name='weight7') 
b7 = tf.compat.v1.Variable(tf.random_normal([20]), name = 'bias7') 
layer7 = tf.nn.relu(tf.matmul(layer6,w7) + b7)

w8 = tf.compat.v1.Variable(tf.random_normal([20,10]), name='weight8')
b8 = tf.compat.v1.Variable(tf.random_normal([10]), name = 'bias8')
layer8 = tf.nn.relu(tf.matmul(layer7,w8) + b8)

w9 = tf.compat.v1.Variable(tf.random_normal([10,1]), name='weight9')
b9 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'bias9')
hypothesis = tf.nn.relu(tf.matmul(layer8, w9) + b9) 



loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=(1e-15))
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(201):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 200== 0:
        print(step, 'loss : ', loss_val,"\n 예측값 : \n", hy_val)
    r2 = r2_score(y_data, hy_val)
    print('R2: ', r2)


sess.close()
