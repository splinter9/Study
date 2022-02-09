import tensorflow as tf
print(tf.__version__)            # 2.7.0
print(tf.executing_eagerly())    # True => 텐서플로2.0 에서는 즉시실행 가능

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())    # Fales

hello = tf.constant("hello world")

print(hello)
