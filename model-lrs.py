import numpy as np
from sklearn.linear_model import LinearRegression

# prima del reshape: [5.1 4.3 3.1 2.6 1.9]
# dopo il reshape: [[5.1] [4.3] [3.1] [2.6] [1.9]]
x = np.array([5.1, 4.3, 3.1, 2.6, 1.9]).reshape((-1, 1))
y = np.array([1.4, 2.2, 3.1, 4.7, 5.4])

# creiamo il modello dall'istanza di LinearRegression
model = LinearRegression()
# Alleniamo il modello
model.fit(x, y)

# coeficente di determinazione (dati di ADDESTRAMENTO)
print('Coeficente di determinazione: %f' % model.score(x, y))

# predizione dei dati
x_pred = np.array([5])
y_pred = model.predict(x_pred.reshape((-1, 1)))

print('predizione: %f = %f' % (x_pred[0], y_pred[0]))
