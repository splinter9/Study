import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)

# sess = tf.Session()
sess = tf.compat.v1.Session()

sess = tf.Session()
print(sess.run([node1, node2])) # node1, node2 : [3.0, 4.0]
print(sess.run(node3))          # node3 :  7.0

