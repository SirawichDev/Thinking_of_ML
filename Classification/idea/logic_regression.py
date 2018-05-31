import numpy as np
import matplotlib.pyplot as plt

#using sigmoid naja
x = np.linspace(-10, 10, num=50)
plt.figure(figsize= (12,3))
plt.plot( x, 1/ (1+np.exp(-x)))
plt.title("Hello Sigmoid")
plt.show()