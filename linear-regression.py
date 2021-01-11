import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

x = np.array([5.1, 4.3, 3.1, 2.6, 1.9])
y = np.array([1.4, 2.2, 3.1, 4.7, 5.4])

res = stats.linregress(x, y)
plt.plot(x, y, 'o', label='statistiche piloti')
plt.plot(x, res.intercept + res.slope*x, 'r', label='retta di regressione')
plt.legend()
plt.show()