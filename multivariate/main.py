import numpy as np
from multivariate.loadData import load
from multivariate.normalise import normalise
from multivariate.gradientDescent import gradientDescent
from multivariate.prediction import predict
import matplotlib.pyplot as plt

#Load data
X,y = load("data.txt")

m,n = X.shape
#Normalise Data
X_norm,mu,sd = normalise(X)

#Add the intercept term
X = np.ones((m, n+1))
X[:,1:n+1] = X_norm

#Hyperparameters
learning_rate = 0.01
total_iters = 400

#Initialise Theta
theta = np.zeros(3);

#Start Training
theta, loss_history = gradientDescent(X, y, theta, learning_rate, total_iters);
input = np.array([[1.00, 1650, 3]])

plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Itterations')
plt.show()


print(theta)
Y = predict(theta,input)
print(Y)