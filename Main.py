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


def distManhattan(a, b):
    dist = sp.spatial.distance.yule(a, b)
    return dist


# TODO BONUS: implementacja metryki Manhattan
# TODO BONUS: implementacja korelacji Persona

class kNN:
    def __init__(self, parametr_k, dane_etykiety_learning):
        self.dane_etykiety_learning = dane_etykiety_learning
        self.dane_learning = dane_etykiety_learning[:, :-1]
        self.etykiety_learning = dane_etykiety_learning[:, -1:]
        self.parametr_k = parametr_k

    def policzDystIsortuj(self, dana_test, fcjaDist):
        y = []
        for x in self.dane_learning:
            y += [[fcjaDist(dana_test, x)]]
        # print(y)
        y = np.append(self.dane_etykiety_learning, y, axis=1)
        z = y[y[:, -1].argsort()]
        return z

    def wybierzMaxZKdanych(self, dane_ety_learn_dyst):
        k_wybranych = dane_ety_learn_dyst[:self.parametr_k, -2:-1]
        unq, cnt = np.unique(k_wybranych, return_counts=True)
        return (k_wybranych[np.argmax(cnt)])

    def etykietaDlaJednej(self, dana_test):
        dane_ety_learn_dyst = self.policzDystIsortuj(dana_test, distEucli)  ##funkcja jako parametr
        return self.wybierzMaxZKdanych(dane_ety_learn_dyst)

    def predict(self, dane_test):  # zwraca etykiety
        """:rtype: list"""
        y = 1
        for x in dane_test:
            y = np.vstack([y, [self.etykietaDlaJednej(x)]])
        y = y[1:]
        return y

    def score(self, dane_test, etykiety_test, fcjaDist):  # zwraca współczynnik poprawnych rozpoznań
        przydzielone_etykiety = self.predict(dane_test)
        a = 0
        b = 0
        for x in przydzielone_etykiety:
            if (x == etykiety_test[a]):
                b += 1
            a += 1
        return b / (a)  # jakiej wielkosci powinno być a?

        # dane_etykiety_learning = importer('iris.data.learning')
        # dane_etykiety_test = importer('iris.data.test')

        # dane_test = dane_etykiety_test[:, :-1]
        # etykiety_test = dane_etykiety_test[:, -1:]

        # nowa = kNN(1, dane_etykiety_learning)
# print(nowa.predict(dane_test, distEucli))

    # print("Procent poprawnych rozpoznań: ")
    # print(nowa.score(dane_test, etykiety_test, distEucli))

    #print(dane_etykiety_learning[:2])
