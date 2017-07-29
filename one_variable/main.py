import numpy as np
from one_variable.gradientDescent import gradientDescent
from one_variable.prediction import predict
import matplotlib.pyplot as plt

from one_variable.loadData import load

X,y = load("one_variable/data.txt")

theta = np.zeros(2) #Initialise theta to zero

#Hyperparameters
learning_rate = 0.01
total_iters = 1500

#Start Training
theta,loss_history = gradientDescent(X, y, theta, learning_rate, total_iters)

#Use the trained weights (theta) to predict
Y = predict(theta,35000)
print("Predicted profit for a city with population of 35000: $", Y)

plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Itterations')
plt.show()