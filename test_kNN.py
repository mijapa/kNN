from unittest import TestCase
from Main import *


def distEucli(a, b):
    dist = sp.spatial.distance.euclidean(a, b)
    return dist


class TestKNN(TestCase):
    dane = ((0, 0, 0, 0, 'zero'), (1, 1, 1, 1, "jeden"), (2, 2, 2, 2, "dwa"))
    dane_etykiety_learning = np.array(dane, dtype=object)
    print(dane_etykiety_learning)
    n = kNN(1, dane_etykiety_learning)
    dana_test = np.array((0, 0, 0, 0), dtype=object)
    score = np.vstack(((0.), (2.), (4.)))
    print(score)
    dane_etykiety_score = np.append(dane_etykiety_learning, score, axis=1)
    print(dane_etykiety_score)
    print(dana_test)
    print(n.policzDystIsortuj(dana_test, distEucli), dane_etykiety_score)

    def test_policzDystIsortuj(self):
        dane = ((0, 0, 0, 0, 'zero'), (1, 1, 1, 1, "jeden"), (2, 2, 2, 2, "dwa"))
        dane_etykiety_learning = np.array(dane, dtype=object)
        print(dane_etykiety_learning)
        n = kNN(1, dane_etykiety_learning)
        dana_test = np.array((0, 0, 0, 0), dtype=object)
        score = np.vstack(((0.), (2.), (4.)))
        print(score)
        dane_etykiety_score = np.append(dane_etykiety_learning, score, axis=1)
        print(dane_etykiety_score)
        print(dana_test)
        print(n.policzDystIsortuj(dana_test, distEucli), dane_etykiety_score)
        wynik = (n.policzDystIsortuj(dana_test, distEucli))
        self.assertSequenceEqual(wynik.tolist(), dane_etykiety_score.tolist())

    def test_wybierzMaxZKdanych(self):
        pass

    def test_etykietaDlaJednej(self):
        pass

    def test_predict(self):
        pass

    def test_score(self):
        pass
