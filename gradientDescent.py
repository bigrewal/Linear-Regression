from loss import computeCost
import numpy as np

def gradientDescent(X, y, theta, learning_rate, total_iters):

    I = X[:,1]
    loss_history = np.zeros(total_iters);

    n = y.size

    for i in range(total_iters):
        theta0 = theta[0]
        theta1 = theta[1]

        hyp = theta0 + (theta1 * I)
        loss = computeCost(theta0, theta1, I, y)
        print("Loss: ",loss)
        loss_history[i] = loss

        diff = hyp - y;

        theta0 = theta0 - (learning_rate/n) * np.sum(diff)
        theta1 = theta1 - (learning_rate/n) * np.sum(diff*I)

        theta[0] = theta0
        theta[1] = theta1

    return theta,loss_history