import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def setupData(fData, fDataShuffle):
    data = pd.read_csv(fData, header=0)
    A = data.values
    np.random.shuffle(A)
    np.savetxt(fDataShuffle, A, delimiter=',')

def setWeights(s, fname):
    theta1 = np.random.rand(s[0], s[1])*0.01
    theta2 = np.random.rand(s[1], s[2])*0.01

    b1 = np.zeros(s[1])
    b2 = np.zeros(s[2])

    t = np.append(theta1, b1)
    t = np.append(t, theta2)
    t = np.append(t, b2)
    np.savetxt(fname, t, delimiter=',')

class Neural_Network():
    def __init__(self, s, useFile, fweights):
        self.s = s
        if useFile:
            t = pd.read_csv(fweights, header=-1)
            t = t.values

            self.setParams(s, t)
        else:
            self.theta1 = np.random.rand(self.s[0], self.s[1])
            self.theta2 = np.random.rand(self.s[1], self.s[2])

            self.b1 = np.random.rand(self.s[1])
            self.b2 = np.random.rand(self.s[2])

    def setParams(self, s, t):
        endTheta1 = s[0]*s[1]
        endB1 = endTheta1 + s[1]
        endTheta2 = endB1 + s[1]*s[2]
            
        self.theta1 = t[:endTheta1].reshape(s[0], s[1])
        self.b1 = t[endTheta1:endB1].reshape(s[1])
        self.theta2 = t[endB1:endTheta2].reshape(s[1], s[2])
        self.b2 = t[endTheta2:].reshape(s[2])

    def getParams(self):
        self.t = np.append(self.theta1, self.b1)
        self.t = np.append(self.t, self.theta2)
        self.t = np.append(self.t, self.b2)
        return self.t

    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z))
    def activation(self, fnc, Z):
        if fnc=='sigmoid':
            return self.sigmoid(Z)
        if fnc=='tanh':
            return np.tanh(Z)
        if fnc=='relu':
            return np.maximum(0, Z)
        return np.maximum(0.01*Z, Z)
    def dactivation(self, fnc, Z):
        if fnc=='sigmoid':
            return self.sigmoid(Z)*(1-self.sigmoid(Z))
        if fnc=='tanh':
            return 1-(np.tanh(Z)**2)
        if fnc=='relu':
            return 1*(Z < 0)
        return 0.01*(Z<0) + 1*(Z>=0)

    def forward(self, X):
        self.Z2 = np.dot(X, self.theta1) + self.b1
        self.a2 = self.activation('sigmoid', self.Z2)

        self.Z3 = np.dot(self.a2, self.theta2) + self.b2
        self.a3 = self.sigmoid(self.Z3)
        return self.a3

    def backprop(self, X, Y, lam=0):
        self.delta3 = (self.a3 - Y) / len(X)
        self.delta2 = np.dot(self.delta3, self.theta2.T) * self.dactivation('sigmoid', self.Z2)

        self.dJdt2 = np.dot(self.a2.T, self.delta3) + lam*self.theta2/len(X)
        self.dJdt1 = np.dot(X.T, self.delta2) + lam*self.theta1/len(X)

        self.dJdb2 = np.sum(self.delta3, axis=0, keepdims=True)
        self.dJdb1 = np.sum(self.delta2, axis=0, keepdims=True)

    def J(self, X, Y, lam):
        self.forward(X)
        self.term1 = np.dot(Y.T, np.log(self.a3))
        self.term2 = np.dot(1-Y.T, np.log(1-self.a3))
        self.reg = lam*(np.sum((self.theta1**2)) + np.sum((self.theta2**2)))/2
        return (-np.sum(self.term1 + self.term2) + self.reg)/len(Y)
        
        
    def gradientDescent(self, X, Y, iters, alpha, lam=0):
        self.jh = np.zeros(iters)
        self.jDev = np.zeros(iters)
        for i in range(iters):
            self.forward(X)
            self.backprop(X, Y, lam)

            self.theta1 = self.theta1 - alpha * self.dJdt1
            self.theta2 = self.theta2 - alpha * self.dJdt2

            self.b1 = self.b1 - alpha * self.dJdb1
            self.b2 = self.b2 - alpha * self.dJdb2

            self.errTrain = self.J(X, Y, lam)
            self.errDev = self.J(XDev, YDev, lam)
            if i>0:
                if self.jh[i-1] < self.errTrain:
                    print('alpha overshoot')
                    return
            self.jh[i] = self.errTrain
            self.jDev[i] = self.errDev
        plt.plot(self.jh)
        plt.plot(self.jDev)
        plt.title(alpha)
        plt.show()

    def accuracy(self, X, Y):
        self.forward(X)
        self.YP = np.array([0 if label <0.5 else 1 for label in self.a3]).reshape((len(Y), 1))
        self.misclassifyErr = np.mean(self.YP == Y)*100
        return self.misclassifyErr

neurons = 26
fweights = 'hWeights' + str(neurons) + '.txt'
dataFile = 'heart.csv'
dataTXT = 'heart.txt'

lam=1
for j in range(neurons, neurons+1, 1):
    s = [13, j, 1]
    print('Architecture s=', s)
    n = s[0]
    target = s[2]
    my_file = Path(dataTXT)
    if not my_file.is_file():
        print('file ', dataTXT, ' not exist, generating file ...')
        setupData(dataFile, dataTXT)
    my_file = Path(fweights)
    if not my_file.is_file():
        print('New weights, generating weights ...')
        setWeights(s, fweights)
    data = pd.read_csv(dataTXT, header=-1)
    A = data.values
    X = A[:, :-1]
    Y = A[:, -1:]
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    trainSize = int(X.shape[0]*0.6)
    testSize = devSize = int(X.shape[0]*0.2)
    endDev = trainSize+devSize
    Xtrain = X[:trainSize]
    XDev = X[trainSize:endDev]
    Xtest = X[endDev:]

    Ytrain = Y[:trainSize]
    YDev = Y[trainSize:endDev]
    Ytest = Y[endDev:]

            
    N = Neural_Network(s, True, fweights)
    alpha = [1]
    t = N.getParams()
    minJDev = 100
    for i in alpha:    
        N.gradientDescent(Xtrain, Ytrain, 800, i, lam)
        ErrDev = N.J(XDev, YDev, lam)
        ErrTrain = N.jh[-1]
        ErrTest = N.J(Xtest, Ytest, lam)
        misErrTrain = N.accuracy(Xtrain, Ytrain)
        misErrDev= N.accuracy(XDev, YDev)
        misErrTest = N.accuracy(Xtest, Ytest)
        print('alpha=', i, '  Error Train: ', ErrTrain, '  Dev = ', ErrDev, ' Test=', ErrTest)
        print('alpha=', i, '  miss classification Train: ', misErrTrain, '  Dev = ', misErrDev, ' Test=', misErrTest)
        if  ErrDev < minJDev:
            minJDev = ErrDev
            minJTrain = ErrTrain
            minAlpha = i
        N.setParams(s, t)
