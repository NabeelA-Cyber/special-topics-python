import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Nabeel Alkhatib\\Desktop\\college stuff\\Special Topics\\Python 2\\ex2data1.txt", names =['x1', 'x2', 'y'])

x = data [['x1', 'x2']]
y = data [['y']]

print(data.head())

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def h(theta,x):
    return sigmoid(np.dot(x, theta))

def normalize(x):
    MuX = x.mean (axis = 0)
    stdX = x.std (axis = 0)
    x_norm = (x-MuX)/ stdX
    return x_norm,MuX,stdX

def J(theta, x, y):
    term1 = np.dot(y.T, np.log(h(theta, x)))
    term2 = np.dot(1-y.T, np.log(1-h(theta, x)))
    return np.sum(-(term1 + term2) / len(x))

def gradientDescent (theta, x, y, alpha, iters):
    m = len(x)
    J_history=[]
    for i in range(iters):
        theta = theta - alpha/m * np.dot(x.T, (h(theta, x) -y))
        J_history.append(J(theta, x, y))

        return theta, J_history

XA = np.array(x)
y = np.array(y)

XA, MuX, stdX = normalize(XA)
m, n = XA.shape

XA = np.append(np.ones((m,1)), XA, axis =1)

alpha = 1.5
iters = 1000
theta = np.zeros((n+1,1))

theta, j_histroy = gradientDescent(theta, XA, y, alpha, iters)

print(theta)

admitted = data[data['y'] == 1]
nonAdmitted = data[data['y'] == 0]

A = np.array(admitted[['x1', 'x2']])
B = np.array(nonAdmitted [['x1', 'x2']])

A = (A - MuX)/ stdX
B = (B - MuX)/ stdX

x_values = np.array ([np.min(XA[:,1]), np.max(XA[:,1])])

x2 = -(theta[0] + theta[1]* x_values) / theta[2]

plt.scatter(A[:,0], A[:,1], marker = '+', label = 'admitted')
plt.scatter(B[:,0], B[:,1], marker = 'x', label = 'nonadmitted')
plt.legend()

plt.plot(x_values, x2)
plt.show()
