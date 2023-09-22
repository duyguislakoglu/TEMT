import torch
import numpy as np

class PositionalEncoder:
    def __init__(self, min, max, dimension):
        self.min = min
        self.max = max
        self.time_vectors = getPositionEncoding(max-min+1, dimension)

    def __call__(self, timestamp):
        # return np.random.rand(3,2)
        return self.time_vectors[timestamp-self.min]

# Code taken from machinelearningmastery.com
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
