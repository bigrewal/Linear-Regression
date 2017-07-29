import numpy as np

def predict(theta, X):
    return np.transpose(theta).dot(np.transpose(X))