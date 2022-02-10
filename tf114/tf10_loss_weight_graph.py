import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
w =tf.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y))

w_history=[]
loss_history=[]

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("=========================== w history =========================")
print(w_history)
print("=========================== loss history ======================")
print(loss_history)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import numpy as np
from numpy.core.fromnumeric import shape


font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=18)

plt.plot(w_history, loss_history)
plt.xlabel('웨이트', font=fontprop)
plt.ylabel('로스', font=fontprop)
plt.title('선생님 만세', font=fontprop)
plt.show()
