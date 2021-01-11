import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5.1, 4.3, 3.1, 2.6, 1.9]).reshape((-1, 1))
y = np.array([1.4, 2.2, 3.1, 4.7, 5.4])

model = LinearRegression().fit(x, y)

x = np.array([6]).reshape((-1, 1))
y_pred = model.predict(x)

print('minuti previsti:', y_pred)