from unittest import TestCase
from Main import distEucli


class TestDistEucli(TestCase):
    def test_distEucli(self):
        self.assertEqual(distEucli(1, 1), 0, "odległosć od samego siebie")
        self.assertEqual(distEucli(0, 1), 1, "odległość 1 od 0")
        self.assertEqual(distEucli([0, 10], [0, 10]), 0, "wektor")
        self.assertEqual(distEucli([1, 10], [1, 0]), 10, "wektor")
