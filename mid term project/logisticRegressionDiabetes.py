## Replace all ... with Python code/codes that satisfy the comment.

## Import the libraries: numpy, pandas, matplotlob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## define you sigmoid function:
def sigmoid(g):
    return 1. / (1. + np.exp(-g))

#define you hypothesis function h for logistic regression:
def h(theta, W):
    return sigmoid(np.dot(W,theta))

## define the error cost function J for logistic regression that return a single float number
def J (theta, W,Y,lam = 0):
    term1 = np.dot(Y.T, np.log(h(theta, W)))
    term2 = np.dot(1-Y.T, np.log(1-h(theta, W)))
    reg =  lam/len(W) * np.sum(theta**2)
    return np.sum(-(term1 + term2) / len(W)) + reg

## define one iteration of gradient descent. This is done for you.
def gradOne(theta, x, y, lam = 0):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    grad = 1./m * np.dot(x.T, (h(theta, x) - y)) + lam/m * theta
    grad[0] = grad[0] - lam/m*theta[0]
    return grad.flatten()

## define the gradient descent function:
def gradientDescent(theta, X, Y, alpha, iters, lam=0):
    m = len(X)
    J_history=[]
    for i in range(iters):
        theta = theta - alpha/m * np.dot(X.T, (h(theta, X) - Y))
        J_history.append(J(theta, X, Y))

    return theta, J_history

## define normlaize features function that returns normalized x, mean and standard deviation
def normalize(X):
    MuX = X.mean (axis = 0)
    stdX = X.std (axis = 0)
    x_norm = (X-MuX)/ stdX
    return x_norm,MuX,stdX
        
## Read using pandas the file binary.csv        
data = pd.read_csv("C:\\Users\\Nabeel Alkhatib\\Desktop\\college stuff\\Special Topics\\mid term project\\diabetes.csv", header = 0,names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y'])

## Extracting features and target
X = data [['x1','x2','x3','x4','x5','x6','x7','x8']]
Y = data [['y']]
################################################
# set the polynomial degree, you can start with 1.
degree = 1

# convert X and Y into arrays:
X1 = np.array(X)
YA = np.array(Y)

## convert target from nominal values of 'tested-positive' and 'tested-negative' to 0 and 1
YA = np.array([1 if label == 'tested_positive' else 0 for label in YA]).reshape((len(X1), 1))

### Normalize:
MuX = X1.mean(axis =0)
stdX = X1.std(axis = 0)
XA = (X1-MuX)/stdX

## Add ones to your feature matrix
XA = np.append(np.ones((len(X1),1)),XA, axis =1) 
############################
## Initialize:
## initialize m and n where m is the number of rows and n is the number of features.
m,n = XA.shape
              
## Initialize theta with zeros
theta = np.zeros((n,1))

## initialize: alpha, iters
alpha = 1
iters = 100
lam = 0.1
## Running Gradient Descent Algorithm
theta, J_history= gradientDescent(theta, XA, YA, alpha, iters, lam)
print("Mean Square Error=", J_history[-1])

### Drawing the cost function J(theta)
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('$J(\Theta)$')
plt.title("Cost function using gradient descent")
plt.show()
#######################################

## Prediction:
## Prediction a test data of: [8, 155, 66, 45, 0, 21.6, 0.627, 34]
xp = np.array([[8. ,155. ,66. ,45. ,0. ,21.6 ,0.627 ,34.]])
## normalize test data
xp_test = (xp-MuX)/stdX

## add ones
xp_test = np.append(np.ones((len(xp_test),1)),xp_test, axis =1)

## predict test data xp_test
yp = 1 if h(theta, xp_test) >= 0.5 else 0

## print result:
print ('****** Prediction: ', yp, ' with probability: ', h(theta, xp_test))

YP = h(theta, XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)
###########

### using scipy.optimize library functions:
import scipy.optimize as opt
initial_theta = np.zeros((n, 1))

def prediction(theta, X):
    return h(theta, X)

print("\n***** Begining optimize library ******** ")

## fmin_cg
print('**** fmin_cg: ****')
result = opt.fmin_cg(f=J, x0=initial_theta, fprime=gradOne, args=(XA, YA, lam), maxiter = 300, disp=False)
## Prediction for binary.csv:
yp = prediction(result, xp_test)
print ('Prediction: ', yp, ' with probability: ', h(result, xp_test))
YP = h(result, XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)

## fmin_bfgs:
print('\n**** fmin_bfgs: ****')
result = opt.fmin_bfgs(f=J, x0=initial_theta, fprime=gradOne, args=(XA, YA, lam), maxiter = 300, disp=False)
## Prediction for binary.csv:
yp = prediction(result, xp_test)
print ('Prediction: ', yp, ' with probability: ', h(result, xp_test))
YP = h(result, XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)


## fmin_l_bfgs_b
print('\n**** fmin_l_bfgs_b: ****')
result = opt.fmin_l_bfgs_b(J, x0=initial_theta, fprime=gradOne, args=(XA, YA, lam), maxiter = 300, disp=False)
yp = prediction(result[0], xp_test)
print ('Prediction: ', yp, ' with probability: ', h(result[0], xp_test))
YP = h(result[0], XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)


## fmin_tnc:
print('\n**** fmin_tnc: ****')
result = opt.fmin_tnc(J, x0=initial_theta, fprime=gradOne, args=(XA, YA, lam), disp=False)
## Prediction for binary.csv:
yp = prediction(result[0], xp_test)
print ('Prediction: ', yp, ' with probability: ', h(result[0], xp_test))
YP = h(result[0], XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)

#### sklearn library:
print ('\nUsing sklean library: ')
from sklearn.linear_model import LogisticRegression
r = LogisticRegression(solver='liblinear')
r.fit(XA, YA.flatten())
Yp = r.predict(XA)
YP = np.array([1 if label>=0.5 else 0 for label in YP]).reshape((len(YP), 1))
Accuracy = np.mean(YP == YA)*100
print("Accuracy = ", Accuracy)

###### END

