import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

t1 = time.time()

c = np.dot(a,b)

t2 = time.time()

print("victored form time=", (t2-t1)*1000)

c = 0

t1 = time.time()
for i in range(1000000):
    c = c + a[i]*b[i]

t2 = time.time()
print("for loop time=", (t2-t1)*1000)
