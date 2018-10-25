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
        pass

    def predict(self, dane_test, distEucli):  # zwraca etykiety
        # print(dane_test)
        print(distEucli(1, 2))
        pass

    def score(self, dane_test, etykiety_test):  #zwraca współczynnik poprawnych rozpoznań
        pass


dane_etykiety = importer('iris.data.learning')
dane_etykiety_x = dane_etykiety[0, :-1]

print(dane_etykiety_x)
dane = dane_etykiety[:, :-1]
print(dane)
etykiety = dane_etykiety[:, -1:]
print(etykiety)

dist = distEucli(dane_etykiety[0, :-1], dane_etykiety[1, :-1])
print(dist)

nowa = kNN(2, dane_etykiety)
nowa.predict(dane_etykiety[:-1], distEucli)
