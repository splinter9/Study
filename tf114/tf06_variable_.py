import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32) # x는 2라는 선언
y = tf.Variable([2], dtype=tf.float32) # y는 2라는 선언

init = tf.compat.v1.global_variables_initializer()  #초기화 시킨다
sess.run(init)

print('잘나오니?', sess.run(x))

