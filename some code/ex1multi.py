### Replace each '...' with the required code line/lines in python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Helpful functions:

## Pause function to stop until use press anykey
def pause():
    programPause = input("Press the <ENTER> key to continue...")

## Read data file ex1data2.csv store it in data variable as DataFrame
data =pd.read_csv('C:\\Users\\Nabeel Alkhatib\\Desktop\\college stuff\\Special Topics\\ex1data2.csv', header = 0)

## Extract data X and Y. Note X contains two columns: size and bedroom. Y contains the prices column
X = data[['size','bedroom']] 
Y = data[['price']]

print("First 10 examples of X: ")
print(X[0:10])

print("First 10 examples of Y: ")
print(Y[0:10])

pause()

## convert DataFrame X and Y into arrays XArray and YArray
XArray = np.array(X)
YArray = np.array(Y)

## Part 1: Feature Normalization:
## -------------------------------------
print("Normalize features:")

## compute mean and standard deviation for each column:
## You may implement this using for loop or manual
## Note: XArray.ndim return the number of columns in X
## 
## compute mean and std, then normalized X is (X-mean)/std as in slide 18 of multiple features
## Compute mean and std for each column
MuX = XArray.mean(axis = 0)
stdX= XArray.std(axis = 0)

XArray = (XArray-MuX)/stdX
## compute mean and std for Y column
MuY = YArray.mean()
stdY = YArray.std()
## normalzie Y = (Y-MuY)/stdY)
YArray = (YArray-MuY)/stdY
## Add the column of 1's to X amd make XNew
XNew = np.append(np.ones((len(XArray),1)),XArray, axis = 1)

## Part 2: Gradient Descent
## ---------------------------------
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
#
# Hint: At prediction, make sure you do the same feature normalization.
#
print("Running gradient descent...")

## Choose some alpha value
alpha = 1
num_iters = 20 ## keep the number of iteration as 20

## Initialize theta to array([[0.], [0.], [0.]]) and Run Gradient Descent
theta = np.zeros((3,1))

def h(theta, X):
    return np.dot(X, theta)

def J(theta, X, Y):
    return ((h(theta, X)-Y) **2)/2/len(X)

def gradientDescent(theta, X, Y, alpha, iterations):
    m = len(X)
    Jh = np.zeros(iterations, dtype='float')
    for i in np.arange(iterations):
        # compute theta using gradient descent formula:
        theta = theta - alpha/m * (np.dot(X.T, (h(theta, X) - Y )))
        Jh[i] = np.sum (J(theta, X, Y))   ## keep this line to store J for all iterations in an array
    return theta, Jh   ## return two arrays: theta and Jh

## Call gradient descent function, pass the correct parameters.
theta, J_history = gradientDescent(theta, XNew, YArray, alpha, num_iters)
print(theta)
print('Mean Square Error: ', np.sum (J(theta, XNew, YArray)))

# plot the convergence graph
plt.plot(np.arange(num_iters), J_history, 'g--')
plt.xlabel('iterations')
plt.ylabel('Cost Function J')
plt.show(block=False)
plt.pause(1)

## predict the price of a house of 1650 square ft and 3 bed rooms, using new theta
## 1) set up x as array([1, 1650, 3])
x = np.array ([1., 1650., 3.])
## 2) Normalize x using mean and std:
x[1] = (x[1]-MuX[0])/stdX[0]
x[2] = (x[2] - MuX[1])/stdX[1]
## 3) compute price by calling h(theta, x)
price = h(theta, x)
## 4) re-normalize price: price * stdY + MuY
price = price * stdY + MuY
print("Predicted price of a 1650 sq-ft, 3 bedrooms using gradient descent: ",  price)

pause()

## Part 3: Normal Equations:
## -----------------------------------
print("Solving with normal equatinos...")
def normalEqn(X, Y):
    theta = np.zeros(X.ndim, dtype='float')
	## type in the normal equation in slide 38 of multiple features linear regression
    A = np.dot(X.T, X)
    B = np.linalg.pinv(A)
    theta= np.dot(np.dot(B, X.T),Y)
    return theta

theta = normalEqn(XNew, YArray)
print("Theta computed from the normal equations:")
print(theta)
print('Mean Square Error: ', np.sum(J(theta, XNew, YArray)))

## predict the price of a house of 1650 square ft and 3 bed rooms, using new theta
## 1) set up x as arrat([1, 1650, 3])
x = np.array ([1., 1650., 3.])
## 2) Normalize x using mean and std:
x[1] = (x[1]-MuX[0])/stdX[0]
x[2] = (x[2] - MuX[1])/stdX[1]
## 3) compute price by calling h(theta, x)
price = h(theta, x)
## 4) re-normalize price: price * SigmaY + MuY
price = price * stdY + MuY
print("Predicted price of a 1650 sq-ft, 3 bedrooms using gradient descent: ",  price)



