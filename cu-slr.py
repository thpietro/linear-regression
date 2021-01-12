import math

# dev by pietro agazzi

class SimpleLinearRegression:

    def __init__(self, x, y):
        '''
        il metodo di construtto

        Parameters
        ----------
        x : list or tuple
            variabili caratteristiche x
        y : list or tuple
            variabili target y

        Raises
        ------
        Exception
            se il numero di valori di x e y non combaciano
        '''
        self.x, self.y = x, y

        # salvo nx
        self.data_len = len(x)
        
        # controllo il numero di dati nelle variabili
        if len(x) != len(y):
            raise Exception('non tractable dataset')
    
    def fit(self):
        '''
        calcolo della retta di regressione
        '''

        # calcolo i valori medi
        self.x_media = self.media(self.x)
        self.y_media = self.media(self.y)

        # calcolo delle serie
        self.x_serie, self.y_serie, self.serie = self.series()

        self.angular = self.angular_coefficient()
        self.origin = self.origin()

    def series(self):
        '''
        calcolo della serie x, y

        Returns
        -------
        float
            serie (xi - mx) ** 2
        flaot 
            serie (yi - my) ** 2
        flaot 
            serie xi * yi
        '''

        x_serie = 0
        y_serie = 0
        serie = 0

        for i in range(self.data_len):
            x_serie += (self.x[i] - self.x_media) ** 2
            y_serie += (self.y[i] - self.y_media) ** 2
            serie += self.x[i] * self.y[i]
        
        return x_serie, y_serie, serie

    def angular_coefficient(self):
        '''
        calcolo del coefficente angolare della retta di regressione
        
        Returns
        -------
        flaot
            coeficente angolare della retta 
        '''
        return (self.serie - self.data_len * self.x_media * self.y_media) / self.x_serie
        
    def origin(self):
        '''
        calcolo dell'origine y0 della retta di regressione
        
        Returns
        -------
        flaot
            y0 della retta
        '''
        return self.y_media - self.angular * self.x_media

    def score(self):
        '''
        calcolo del coefficente di correlazione
        
        Returns
        -------
        flaot
            coefficente di correlazione
        '''
        return (self.serie - self.data_len * self.x_media * self.y_media) / (math.sqrt(self.x_serie * self.y_serie)) 
    
    def predict(self, x):
        calc = lambda x : self.angular * x + self.origin
        return [calc(i) for i in x]

            
    @staticmethod
    def media(a):
        return sum(a) / len(a)


model = SimpleLinearRegression(
    [5.4, 4.5, 3.4, 2.3, 1.8],
    [1.0, 2.2, 3.9, 4.5, 5.1]
)
model.fit()

print('coeficente di correlazione: %f' % model.score())
print(model.predict([5]))

