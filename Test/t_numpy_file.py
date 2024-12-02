
import numpy as np

a = np.array([[1,2,3,4],[2,3,4,5]])

# write to .np file
np.save("t_numpy_file.npy", a)

# read/recover from .np file
b = np.load('./t_numpy_file.npy')

print("b = ", b, type(b))

# for write and load multiple ndarray, one can use savez