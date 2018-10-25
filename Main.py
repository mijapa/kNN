import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial as st


def importer(file):
    my_data = pd.read_csv(file)  # panda is faster than numpy
    array = np.array(my_data)
    return array


def distEucli(a, b):
    dist = sp.spatial.distance.euclidean(a, b)
    return dist

class kNN:
    def __init__(self, parametr_k, dane_etykiety_learning):
        self.dane_etykiety_learning = dane_etykiety_learning
        self.dane_learning = dane_etykiety_learning[:, :-1]
        self.etykiety_learning = dane_etykiety_learning[:, -1:]
        self.parametr_k = parametr_k

    def predict(self, dane_test, fcjaDist):  # zwraca etykiety
        for x in dane_test:
            for y in self.etykiety_learning:
                pass



    def score(self, dane_test, etykiety_test):  #zwraca współczynnik poprawnych rozpoznań
        pass

    def wybierzKnajblizszych(self, dane_test, fcjaDist):
        dana_test = dane_test[0]
        for x in self.dane_learning:
            print(fcjaDist(dana_test, x))


dane_etykiety_learning = importer('iris.data.learning')
dane_etykiety_test = importer('iris.data.test')

dane_test = dane_etykiety_test[:, :-1]
etykiety_test = dane_etykiety_test[:, -1:]

nowa = kNN(2, dane_etykiety_learning)
nowa.predict(dane_test, distEucli)
nowa.wybierzKnajblizszych(dane_test, distEucli)
