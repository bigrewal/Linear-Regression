import numpy as np

def load(file):
    data = np.loadtxt(file, delimiter=',')
    m,n = data.shape

    X = data[:,:2]
    y = data[:,n-1]

    return X,y