import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1, 2], [2, 5], [3, 4]])

x = data[:, 0]
y = data[:, 1]

x1 = x.reshape(3, 1)
y = y.reshape(3, 1)

x = np.append(np.ones((len(x1), 1)), x1, axis=1)



def h(theta, x):
    return np.dot(x, theta)
def J(theta, x, y):
    return ((h(theta, x)-y)**2)/2/len(x)
def gradientDescent(theta, x, y, alpha, iter):
    m = len(x)
    for i in range(iter):
        theta = theta - alpha/m * (np.dot(x.T, (h(theta, x) - y)))
    return theta

theta = np.array([[0.], [0.]])
alpha = 0.01
iter = 100

theta = gradientDescent(theta, x, y, alpha, iter)
print('Theta = ', theta)

yp = h(theta, x)
print('yp=', yp)


plt.scatter(x1, y)
plt.plot(x1, yp)
plt.show()

































