import matplotlib.pyplot as plt
import numpy as np

x = np.array([5.1, 4.3, 3.1, 2.6, 1.9])
y = np.array([1.4, 2.2, 3.1, 4.7, 5.4])

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 

plt.plot(x,y, 'x', x, fit_fn(x), 'r')
plt.show()
