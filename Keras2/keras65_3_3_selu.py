# 난 정말 셀루

import numpy as np
import matplotlib.pyplot as plt

def selu(x):
    return (np.exp(x) - 1) * 1.6732632423543772848170429916717

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()

