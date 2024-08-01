import numpy as np
import pandas as pd

data = pd.read_csv ('test.csv', header =0, names=['x1','x2','y'])


X = data [['x1','x2']]
Y = data [[ 'y']]

print(X)
print(Y)

XA = np.array(X)
YA = np.array(Y)

##Normalize:
MuX = XA.mean(axis = 0)
stdX = XA.std(axis = 0)

XA = (XA - MuX) / stdX

XA = np.append(np.ones((len(XA),1)),XA, axis =1)
YA = np.array([1 if label == 'high' else 0 for label in YA ])
YA = YA.reshape(2,1)

m,n = XA.shape

theta = np.zeros((n,1))
alpha = 0.1
iters = 100
lam = 0.1
def sigmoid(z):
    return 1./(1.+np.exp(-z))
def h(theata, X):
    return sigmoid(np.dot( X, theta))
def J(theta, X, Y, lam=0):
    term1 = np.dot(Y.T, np.log(h(theta,X)))
    term2 = np.dot(1-Y.T, np.log(1-h(theta,X)))
    reg = lam/len(X) * np.sum(theta**2)
    return np.sum(-(term1 + term2) / len(X)) + reg
def Jprime(theta,X,Y,lam=0):
    return np.dot(X.T,h(theta, X) - Y)/ len(X) + lam/len(X) * theta
def gradientDescent(theta, X, Y, alpha, iters, lam=0):
    Jh = []
    for i in range(iters):
        d = Jprime(theta, X, Y, lam)
        theta = theta - alpha * d
        Jh.append(J(theta,X,Y))
        return theta, Jh
    
theta1, J_history = gradientDescent(theta, XA, YA, alpha, iters)
print(theta1)
print(J_history[-1])
