import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data = np.array([[1, 2], [2, 5], [3, 4]])
data = pd.read_csv("C:\\Users\\oelkhat\\Desktop\\Loyola university\\Spring 2019\\Machine Learning\ml\\MyLectures\\Datasets\\web_traffic1.csv", names=['x', 'y'])

x = data[['x']]
y = data[['y']]

x1 = np.array(x)    # x.values
y = np.array(y)    # y.values

x1 = (x1 - x1.mean()) / x1.std()   ## mean-std normalization

#x1 = x.reshape(3, 1)
#y = y.reshape(3, 1)
#plt.scatter(x1, y)
#plt.show()


x = np.append(np.ones((len(x1), 1)), x1, axis=1)
x = np.append(x, x1**2, axis=1)
x = np.append(x, x1**3, axis=1)



def h(theta, x):
    return np.dot(x, theta)
def J(theta, x, y):
    return ((h(theta, x)-y)**2)/2/len(x)
def gradientDescent(theta, x, y, alpha, iter):
    m = len(x)
    for i in range(iter):
        theta = theta - alpha/m * (np.dot(x.T, (h(theta, x) - y)))
        #print("Mean Square Error=", np.sum(J(theta, x, y)))
    return theta

theta = np.array([[0.], [0.], [0.], [0.]])
alpha = 0.01
iter = 1000

theta = gradientDescent(theta, x, y, alpha, iter)
print('Theta = ', theta)

x2 = np.arange(-2, 2, 0.01)
x2 = x2.reshape(len(x2), 1)
x3 = np.append(np.ones((len(x2), 1)), x2, axis=1)
x3 = np.append(x3, x2**2, axis=1)
x3 = np.append(x3, x2**3, axis=1)
yp = h(theta, x3)


plt.scatter(x1, y)
plt.plot(x2, yp, color='red')
plt.show()

































