import numpy as np

X = np.array([[3, 5], [5, 1], [10, 2]], dtype=float)
Y = np.array([[75], [82], [93]], dtype=float)

class Neural_Network():
    def __init__(self):
        self.input = 2
        self.hidden = 3
        self.output = 1

        self.theta1 = np.random.rand(2, 3)
        self.theta2 = np.random.rand(3, 1)

        self.b1 = np.random.rand(1, 3)
        self.b2 = np.random.rand(1, 1)

    def forward(self, X):
        self.Z2 = np.dot(X, self.theta1)
        self.a2 = self.sigmoid(self.Z2+self.b1)

        self.Z3 = np.dot(self.a2, self.theta2)
        self.a3 = self.sigmoid(self.Z3+self.b2)

        return self.a3
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z))
    
N = Neural_Network()
h = N.forward(X)
print('h=', h)








