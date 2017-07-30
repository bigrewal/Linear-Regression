import numpy as np

def computeCost(theta, X, y):
    hyp = theta[0] + (theta[1]*X)  #Hypothesis function
    n = y.size
    total_squared_error = (hyp - y)**2
    loss = np.sum(total_squared_error)/(2*n)

    return loss