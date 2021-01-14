''' Pietro Agazzi
Regressione Lineare Semplice
'''

import numpy as np
from math import sqrt

class SimpleLinearRegression:

    def fit(self, X, y):
        ''' Addestramento dei dati.

        Parameters
        ----------
        X : array
          features
        y : array
          target
        '''
        self.angularity = self.__angularity(X, y)
        self.intercepts = self.__intercepts(X, y, self.angularity)
        
    def score(self, x, y):
        ''' Ottiene il coefficente di correlazione

        Parameters
        ----------
        X : array
          features
        y : array
          target

        Returns
        -------
        coefficent : float
        '''
        return self.__coefficents(x, y)

    def predict(self, x):
        ''' Predici i target

        Parameters
        ----------
        x : array
          features

        Returns
        -------
        target : array
        '''
        calc = lambda x : self.angularity * x + self.intercepts
        return [calc(i) for i in x]

    def __intercepts(self, x, y, angularity):
        ''' Calcolare l'intercetta
        return self.__media(y) - intercepts * self.__media(x)

        Parameters
        ----------
        x : array
        y : array
        angularity : float

        Returns
        -------
        q : float
        '''
        return self.__media(y) - angularity * self.__media(x)
        
    def __dev(self, a):
        ''' dev(a) -> Σ (a_i - am)^2

        Parameters
        ----------
        a : array
          features

        Returns
        -------
        dev(x) : float
        '''
        a_m = self.__media(a)
        return sum([(n - a_m) ** 2 for n in a])

    def __codev(self, a, b):
        ''' codev(a, b) -> Σ (a_i - am) (b_i - bm)

        Parameters
        ----------
        a : array
        b : array

        Returns
        -------
        codev(a, b) : float
        '''
        a_m = self.__media(a)
        b_m = self.__media(b)
        return sum([(a[i] * b[i]) for i in range(len(a))])

    def __angularity(self, x, y):
        ''' Calcola il coefficente angolare
        m = codev(x, y) - n xm ym / dev(x)

        Parameters
        ----------
        x : array
        y : array

        Returns
        -------
        m : float
        '''
        return (self.__codev(x, y) - len(x) * self.__media(x) * self.__media(y)) / self.__dev(x)

    def __coefficents(self, x, y):
        ''' Calcolo il coefficente di correlazione
        r = ( codev(x, y) - n xm ym ) / sqrt ( dev(x) * dev(y) )

        Parameters
        ----------
        x : array
        y : array

        Returns
        -------
        coefficents : float
        '''
        return (self.__codev(x, y) - len(x) * self.__media(x) * self.__media(y)) / sqrt(self.__dev(x) * self.__dev(y))

    def __media(self, a):
        ''' Media

        Parameters
        ----------
        a : array

        Returns
        -------
        media : int
        '''
        return sum(a) / len(a)

model = SimpleLinearRegression()

x = [20, 24, 28, 30, 32, 36]
y = [10, 12, 18, 24, 22, 20]

model.fit(x, y)
print('Coefficente di correlazione %f' % model.score(x, y))
print(model.predict(x))
