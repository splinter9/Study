from asyncio import constants
import tensorflow as tf
print(tf.__version__)

print('hello world')

hello = tf.constant('hello world') #constant 함수의 특징은?
print(hello)

sess = tf.Session()
print(sess.run(hello)) # b'hello world' 