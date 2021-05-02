# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#générer 2 tableaux x et y
x, y = make_regression(n_samples=100, n_features=1, noise=10)

y=y.reshape((y.shape[0]),1) #réécrire les dimensions de y

print("\n Dimension de x : \n", x.shape)
print("\n Dimensions de y : \n", y.shape)

#matrice X
X = np.hstack((x, np.ones(x.shape)))
#print("\n Matrice X : \n \n",X)
print("\n Dimensions de la matrice  X : \n", X.shape)

theta = np.random.randn(2, 1)
print("\n Theta aléatoire : \n", theta)
print("\n Dimensions de theta aléatoire : \n", theta.shape)


#Modèle

def model(X, theta):
    return X.dot(theta)

#print("\n Modèle avec theta aléatoire : \n \n", model(X, theta))

plt.figure()
plt.title("Modèle avec theta aléatoire")
plt.scatter (x, y, label='nuage de points', c="blue")
plt.plot(x, model(X, theta), label='Vecteur avec theta aléatoire', c="red")
plt.xlabel('axe x')
plt.ylabel('axe y')



#Fonction coût

def  cost_function(X, y, theta):
    m = len(y)
    return (1/(1*m)) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta)-y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range (n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history


#ML

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000)
print ("\n Theta final :\n", theta_final)

predictions = model(X, theta_final)

'''
plt.figure()
plt.title("Modèle final avec theta final")
plt.scatter(x,y, label='nuage de points', c='blue')
plt.plot(x, predictions, label='Vecteur avec theta final', c='pink')
plt.xlabel('axe x')
plt.ylabel('axe y')
plt.legend()

plt.show()
'''
#rajouter au plot du vecteur final
plt.title("Modèle final avec theta final")
plt.plot(x, predictions, label='Vecteur avec theta final', c='pink')
plt.legend()


#courbe d'apprentissage
plt.figure()
plt.title("Courbe d'apprentissage")
plt.plot(range(1000), cost_history)

plt.show()

def coef_determination(y, pred):
    u = ((y-pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

print("\n Coeffictient de determination : \n", coef_determination(y, predictions))
