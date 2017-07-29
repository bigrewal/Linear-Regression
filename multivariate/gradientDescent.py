import numpy as np
from multivariate.loss import computeCost
from multivariate.prediction import predict

def gradientDescent(X, y, theta, learning_rate, total_iters):


    loss_history = np.zeros(total_iters);

    n = y.size
    for i in range(total_iters):

        hyp = predict(theta,X)
        loss = computeCost(theta, X, y)
        print("Loss: ",loss)
        loss_history[i] = loss

        diff = np.transpose(hyp) - y;

        theta = theta - np.transpose((learning_rate/n) * (np.transpose(diff).dot(X)))

    return theta,loss_history