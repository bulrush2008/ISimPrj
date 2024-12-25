
import numpy as np

def func(arr):
  print(arr.size)

  for idx in range(arr.size):
    arr[idx] = arr[idx] + 1
  pass

if __name__=="__main__":
  a = np.array([3,4,9,100])

  func(a)

  print(a)