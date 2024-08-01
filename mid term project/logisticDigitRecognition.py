## Replace ... with Python code/codes.
# import libraries
# import numpy
import numpy as np 
# import matplotlib.pyplot
import matplotlib.pyplot as plt
# import scipy.io and scipy.optimize:
import scipy.io as sp
import scipy.optimize as opt

## define sigmoid:
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

## define hypothesis function h:
def h(theta, X):
    return sigmoid(np.dot(X,theta))

## define cost function or error function, add regularization and return single float number
def J(theta, X, Y, lam = 0):
    term1 = np.dot(Y.T, np.log(h(theta, X)))
    term2 = np.dot(1-Y.T, np.log(1-h(theta, X)))
    d = lam/len(X)*(np.sum(theta**2) - theta[0]**2)
    return np.sum(-(term1 + term2) / len(X)) + lam/len(X)*np.sum(theta**2)
## define one iteration of gradient descent with regularization, done for you.
def oneGrad(theta, X, Y, lam=0):
    theta = theta.reshape((X.shape[1], 1))
    temp = np.dot(X.T, h(theta, X) - Y) + theta*lam
    temp[0] = temp[0] - theta[0] * lam
    temp = temp/len(X)
    return temp.flatten()

## define gradient descent with regularization, return theta and J history
def gradDescent(theta, X, Y, alpha, iters, lam = 0):
    m = len(X)
    J_history=[]
    for i in range(iters):
        d = np.dot(X.T, (h(theta, X) - Y)) + lam*theta - lam*theta[0]
        theta = theta - alpha/m * d
        J_history.append(J(theta, X, Y))

        return theta, J_history

def pause():
    programPause = input("Press the <ENTER> key to continue...")

## Read data file: ex3data1.mat
data = sp.loadmat('ex3data1.mat')
## Extract features:
X = data['X']
Y = data['y']

## Display random digits from the data file
_, axarr = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        r = np.random.randint(X.shape[0])  ## get an int random number from 5000 examples
        XA = X[r].reshape((20, 20), order='F') ## get that image from X
        axarr[i,j].imshow(XA)       ## show in sub-figure (i, j)
axarr[i,j].axis('off')  # turn off axis...
plt.show()

### Add one to features X
XA = np.append(np.ones((len(X),1)),X, axis =1) 

## get m: number of data rows, and n: number of features (Should be m=5000 and n=401)
m, n = XA.shape

## define regularization lambda 
lam = 10

# define number of classes:
K = 10

# define your theta:
theta = np.zeros((10, XA.shape[1]))
# define theta for fmin_cg and fmin_bfgs
theta1 = np.zeros((10, XA.shape[1]))
theta2 = np.zeros((10, XA.shape[1]))

# define jumber of iterations and alpha:
iters = 1000
alpha = 4

## One-VS- all classifier:
## Loop for all classes K
for i in range(K):
    digitClass = i if i else 10               # get one class
    print('digit class = ', i)
    YA = Y==digitClass                  # YA is one vs all class
    
    t = theta[i].reshape((n, 1))    # extract theta for that class
    # call gradient descent with paramters: t, XA, YA, alpha, iters, and lam
    t1, J1 = gradDescent (theta[i].reshape((401, 1)), XA, YA, alpha, iters, lam)
    print('Gradient descnet completed')
    
    w = theta[i].reshape((n, 1))
    e1 = opt.fmin_cg(f=J, x0=w, fprime=oneGrad, args = (XA, YA, 0.1), maxiter=300, disp=False)
    theta1[i] = e1

    w = theta[i].reshape((n, 1))
    e2 = opt.fmin_tnc(func=J, x0=w, fprime=oneGrad, args = (XA, YA, 0.1), disp=False)
    theta2[i] = e2[0]
    theta[i] = t1.flatten()

## Predict all
pred1 = np.argmax(XA @ theta.T, axis = 1)
pred1 = [e if e else 10 for e in pred1]
print('mean=', np.mean(pred1 == Y.flatten()) * 100)

pred = np.argmax(XA @ theta1.T, axis = 1)
pred = [e if e else 10 for e in pred]
print('mean=', np.mean(pred == Y.flatten()) * 100)

pred = np.argmax(XA @ theta2.T, axis = 1)
pred = [e if e else 10 for e in pred]
print('mean=', np.mean(pred == Y.flatten()) * 100)


 ## Using sklearn
from sklearn.linear_model import LogisticRegression

r = LogisticRegression(C=10, penalty='l2', solver='liblinear', multi_class='auto')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
r.fit(X,Y.ravel())

pred2 = r.predict(X)
print('Training set accuracy: {} %'.format(np.mean(pred2 == Y.flatten())*100))

