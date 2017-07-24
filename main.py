import numpy as np
import matplotlib.pyplot as plt

from loadData import load
from prediction import predict
from gradientDescent import gradientDescent

X,y = load('data.txt')

theta = np.zeros(2) #Initialise theta to zero
print(theta[0])
print(theta[1])

#Hyperparameters
learning_rate = 0.01
total_iters = 1500

theta,loss_history = gradientDescent(X, y, theta, learning_rate, total_iters)

Y = predict(theta,35000)