import unittest
import numpy as np

from cortio import Cortio

class TestCortio(unittest.TestCase):
    def test_signal_1(self):
        x = Cortio.transform_file('data/speech1.wav')
        y = np.load('tests/fixtures/speech1-crt.npy')
        mse = np.mean((x - y) ** 2)
        self.assertTrue(mse < 1e-12)

if __name__ == '__main__':
    unittest.main()
