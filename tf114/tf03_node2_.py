import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

#실습
#덧셈 node3
#뺄셈 node4
#곱셉 node5
#나눗셈 node6

node3 = node1 + node2
node3 = tf.add(node1, node2)

node4 = node1 - node2
node4 = tf.subtract(node1, node2)


node5 = node1 * node2
node5 = tf.multiply(node1, node2)

node6 = node1 / node2
node6 = tf.divide(node1, node2)


print(node3)
print(node4)
print(node5)
print(node6)

sess = tf.Session()
print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))



'''
# node3 = node1 + node2
node3 = tf.add(node1,node2)
print(sess.run(node3))

# node4 = node1 - node2
node4 = tf.subtract(node1,node2)
print(sess.run(node4))

# node5 = node1 * node2
node5 = tf.multiply(node1,node2)
print(sess.run(node5))

# node6 = node1 / node2
node6 = tf.divide(node1,node2)
print(sess.run(node6))
'''


