import numpy as np
from multivariate.prediction import predict

def computeCost(theta,X,y):
    hyp = predict(theta,X)
    n = y.size
    total_squared_error = (np.transpose(hyp) - y) ** 2
    loss = np.sum(total_squared_error) / (2 * n)

    return loss