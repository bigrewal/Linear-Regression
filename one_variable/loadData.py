import numpy as np

def load(file):
    x,y = np.loadtxt(file, delimiter=',', usecols=(0, 1), unpack=True)
    m = x.size

    X = np.ones((m,2))
    X[:,1] = x

    return X,y