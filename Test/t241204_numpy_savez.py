
import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([1.1, 2.3])

np.savez("multi.npz", a=a, b=b)

f = np.load("multi.npz")
a1 = f['a']
b1 = f['b']

print(a1)
print(b1)