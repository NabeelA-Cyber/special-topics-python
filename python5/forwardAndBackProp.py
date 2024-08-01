import numpy as np

X = np.array([[3, 5], [5, 1], [10, 2]], dtype=float)
Y = np.array([[75], [82], [93]], dtype=float)


class Neural_Network():
    def __init__(self):
        self.s = [2, 3, 1]
        self.theta1 = np.random.rand(self.s[0], self.s[1])
        self.theta2 = np.random.rand(self.s[1], self.s[2])

        self.b1 = np.random.rand(self.s[1])
        self.b2 = np.random.rand(self.s[2])

    def forward(self, X):
        self.Z2 = np.dot(X, self.theta1) + self.b1
        self.a2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.a2, self.theta2) + self.b2
        self.a3 = self.sigmoid(self.Z3)

        return self.a3
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z))

    def backprop(self, X, Y):
        self.delta3 = self.a3 - Y
        self.delta2 = np.dot(self.delta3, self.theta2.T) * self.a2 * (1-self.a2)

        self.dJdt2 = np.dot(self.a2.T, self.delta3)
        self.dJdt1 = np.dot(X.T, self.delta2)

        self.dJdb2 = self.delta3
        self.dJdb1 = self.delta2
    def J(self, Y):
        term1 = np.dot(Y.T, np.log(self.a3))
        term2 = np.dot(1-Y.T, np.log(1-self.a3))
        return -np.sum(term1 + term2)/len(Y)
        
        
    def gradientDescent(self, X, Y, iters, alpha):
        for i in range(iters):
            self.forward(X)
            self.backprop(X, Y)

            print('Iteration: ', i, '  Error=', self.J(Y))
            self.theta1 = self.theta1 - alpha * self.dJdt1
            self.theta2 = self.theta2 - alpha * self.dJdt2

            self.b1 = self.b1 - alpha * self.dJdb1
            self.b2 = self.b2 - alpha * self.dJdb2
            
    
N = Neural_Network()
N.gradientDescent(X, Y, 10, 0.0001)








