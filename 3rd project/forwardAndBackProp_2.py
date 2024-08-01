import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def setupData(fData, fDataShuffle):
    ## function to read data file, shuffle it and save it in a file name fDataShuffle
    ## input: fData: data file name
    ##              fDataShuffle: file name to store shuffled data
    data = pd.read_csv(fData, header=1)
    A = data.values
    np.random.shuffle(A)    ## shuffle the data
    np.savetxt(fDataShuffle, A, delimiter=',')

def setWeights(s, fname):
    ## function to initialize weights of your neural network
    ## input: fname: file name to store the weights of neural network.
    epi1 = (6**0.5) / ((s[0]+s[1])**0.5)
    epi2 = (6**0.5) / ((s[1]+s[2])**0.5)
    theta1 = np.random.rand(s[0], s[1])*2*epi1 - epi1
    theta2 = np.random.rand(s[1], s[2])*2*epi2 - epi2

    b1 = np.zeros(s[1])
    b2 = np.zeros(s[2])

    t = np.append(theta1, b1)
    t = np.append(t, theta2)
    t = np.append(t, b2)
    np.savetxt(fname, t, delimiter=',')

class Neural_Network():
    def __init__(self, s, useFile, fweights):
        ## input: s is the neural network architecutre
        ##              useFile: a boolean variable that use a file to initialize the weights.
        ##              fweights: file name where the weights are stored.
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
        ## function to unroll the theta t from its vector form to theta1, b1, theta2, b2
        ## Input: s is the neural network architecture
        ##               t is the thetas vector
        endTheta1 = s[0]*s[1]
        endB1 = endTheta1 + s[1]
        endTheta2 = endB1 + s[1]*s[2]
            
        self.theta1 = t[:endTheta1].reshape(s[0], s[1])
        self.b1 = t[endTheta1:endB1].reshape(s[1])
        self.theta2 = t[endB1:endTheta2].reshape(s[1], s[2])
        self.b2 = t[endTheta2:].reshape(s[2])

    def getParams(self):
        ## function to roll the theta1, b1, theta2, b2 to a big vector t
        self.t = np.append(self.theta1, self.b1)
        self.t = np.append(self.t, self.theta2)
        self.t = np.append(self.t, self.b2)
        return self.t

    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z))
    def activation(self, fnc, Z):
        ## function to compute the activation function of a neuron
        ## input: fnc is the activation function name used
        ##              Z is the value to compute the activation function on
        if fnc=='sigmoid':
            return self.sigmoid(Z)
        if fnc=='tanh':
            return np.tanh(Z)
        if fnc=='relu':
            return np.maximum(0, Z)
        return np.maximum(0.01*Z, Z)
    def dactivation(self, fnc, Z):
        ## function to compute the derivative of the activation function of a neuron
        ## input: fnc is the activation function name used
        ##              Z is the value to compute the derivative of the activation function on
        if fnc=='sigmoid':
            return self.sigmoid(Z)*(1-self.sigmoid(Z))
        if fnc=='tanh':
            return 1-(np.tanh(Z)**2)
        if fnc=='relu':
            return 1*(Z < 0)
        return 0.01*(Z<0) + 1*(Z>=0)

    def forward(self, X):
        ## function to compute the forward propagation
        ## input: X is the input data, a matrix m x n
        self.Z2 = np.dot(X, self.theta1) + self.b1
        self.a2 = self.activation('sigmoid', self.Z2)

        self.Z3 = np.dot(self.a2, self.theta2) + self.b2
        self.a3 = self.sigmoid(self.Z3)
        return self.a3

    def backprop(self, X, Y, lam=0):
        ## a function to compute the back propagation
        ## Input: X: input data
        ##              Y: target value
        ##              lam: lambda (regularization parameter)
        self.delta3 = (self.a3 - Y) / len(X)
        self.delta2 = np.dot(self.delta3, self.theta2.T) * self.dactivation('sigmoid', self.Z2)

        self.dJdt2 = np.dot(self.a2.T, self.delta3) + lam*self.theta2/len(X)
        self.dJdt1 = np.dot(X.T, self.delta2) + lam*self.theta1/len(X)

        self.dJdb2 = np.sum(self.delta3, axis=0, keepdims=True)
        self.dJdb1 = np.sum(self.delta2, axis=0, keepdims=True)

    def J(self, X, Y, lam):
        ## function to compute the error:
        ## input: X: input data
        ##              Y: target value
        ##             lam: lambda (regularization parameter)
        self.forward(X)
        self.term1 = np.dot(Y.T, np.log(self.a3))
        self.term2 = np.dot(1-Y.T, np.log(1-self.a3))
        self.reg = lam*(np.sum((self.theta1**2)) + np.sum((self.theta2**2)))/2
        return (-np.sum(self.term1 + self.term2) + self.reg)/len(Y)
        
        
    def gradientDescent(self, X, Y, iters, alpha, lam=0):
        ## function to compute the gradient descent algorithm:
        ## input: X: input data
        ##              Y: target value
        ##             iters: numbe of iterations for gradient descent
        ##             alpha: learning rate.
        ##             lam: lambda (regularization parameter)
        ## Output (return value): 0 means graident descent overshoot
        ##                                               1 means gradient descent completed its iterations successfully.
        self.jh = np.zeros(iters)
        self.jDev = np.zeros(iters)
        for i in range(iters):
            ## call forward propagation
            self.forward(X)
            ## call back propagation
            self.backprop(X, Y, lam)

            ## Update weights: theta1, theta2, b1 and b2
            self.theta1 = self.theta1 - alpha * self.dJdt1
            self.theta2 = self.theta2 - alpha * self.dJdt2

            self.b1 = self.b1 - alpha * self.dJdb1
            self.b2 = self.b2 - alpha * self.dJdb2

            self.errTrain = self.J(X, Y, lam)   ## the error for the current iteration on data X
            self.errDev = self.J(XDev, YDev, lam) ## error of the current iteration of the dev data set
            if i>0:
                if self.jh[i-1] < self.errTrain: ## if the last error is less than the current error
                    print('alpha overshoot')
                    return 0
            self.jh[i] = self.errTrain
            self.jDev[i] = self.errDev
        return 1

    def accuracy(self, X, Y):
        ## function to compute the correct classifications
        ## input: X,Y: input data and target value.
        self.forward(X)
        self.YP = np.array([0 if label <0.5 else 1 for label in self.a3]).reshape((len(Y), 1))
        self.misclassifyErr = np.mean(self.YP == Y)*100
        return self.misclassifyErr

neurons = 26  ## number of neurons in the hidden layer
fweights = 'hrtWeights' + str(neurons) + '.txt' ## name of the file name that stores (or to store) weights.
dataFile = 'pd_speech_features.csv' ## name of data file
dataTXT = 'heart.txt' ## name of file that stores shuffled data.

lam=0 ## regularization parameter
alpha = [0.15]  ## learning rate
for j in range(neurons, neurons+1, 1):
    minJDev = minJTrain = 100
    minAlpha=alpha[0]
    s = [754, j, 1] ## neural network architecture
    print('Architecture s=', s)
    n = s[0]
    target = s[2]
    my_file = Path(dataTXT)
    if not my_file.is_file(): ## check the existence of the file that stores the shuffled data
        print('file ', dataTXT, ' not exist, generating file ...')
        setupData(dataFile, dataTXT) ## if file not exist then generate the shuffled data file.
    my_file = Path(fweights)
    if not my_file.is_file(): ## check the existence of the file that stores the weights of the neural network
        print('New weights, generating weights ...')
        setWeights(s, fweights) ## if file not exist then generate the weight file.
    data = pd.read_csv(dataTXT, header=-1) ## read the shuffled data
    A = data.values
    X = A[:, :-1]
    Y = A[:, -1:]
    X = (X - X.mean(axis=0)) / X.std(axis=0) ## normalize the data using mean and standard deviation

    ## split the data into 60% training, 20% dev and 20% test set
    trainSize = int(X.shape[0]*0.6) 
    testSize = devSize = int(X.shape[0]*0.2)
    endDev = trainSize+devSize
    Xtrain = X[:trainSize]
    XDev = X[trainSize:endDev]
    Xtest = X[endDev:]

    Ytrain = Y[:trainSize]
    YDev = Y[trainSize:endDev]
    Ytest = Y[endDev:]

            
    N = Neural_Network(s, True, fweights)   ## initialize your neural network class
    t = N.getParams() ## roll theta1, theta2, b1, b2 into a big vector t
    bestT = t
    for i in alpha:    
        if N.gradientDescent(Xtrain, Ytrain, 180, i, lam):  ## call graident descent algorithm
            ErrDev = N.J(XDev, YDev, lam)   ## compute dev set error
            ErrTrain = N.jh[-1] ## compute train set error
            ErrTest = N.J(Xtest, Ytest, lam) ## compute test set error
            AccTrain = N.accuracy(Xtrain, Ytrain) ## Accuracy of train
            AccDev= N.accuracy(XDev, YDev) ## Accuracy of dev
            AccTest = N.accuracy(Xtest, Ytest) ## Accuracy of test
            print('alpha=', i, '  Error Train: ', ErrTrain, '  Dev = ', ErrDev, ' Test=', ErrTest)
            print('alpha=', i, '  Accuracy: Train: ', AccTrain, '  Dev = ', AccDev, ' Test=', AccTest)
            plt.plot(N.jh)
            plt.plot(N.jDev)
            plt.title(alpha)
            plt.show()
            if  ErrDev < minJDev:  ## save the minimum dev set error
                minJDev = ErrDev
                minJTrain = ErrTrain
                minAlpha = i
                bestT = t
        N.setParams(s, t) ## initialize your theta1, theta2, b1 and b2  with t

N.setParams(s, bestT)
print('alpha=', minAlpha, '  Error Train: ', minJTrain, '  Dev = ', minJDev, ' Test=', N.J(Xtest, Ytest, lam))

        
