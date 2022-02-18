# 난 정말 리키렐루

import numpy as np
import matplotlib.pyplot as plt

def reakyrelu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = reakyrelu(x)

plt.plot(x, y)
plt.grid()
plt.show()