
import numpy as np

a = [1.0, 2.1, 3.5]
b = [2.0, 3.2, 5.1]

anp = np.array(a); bnp = np.array(b)

e = 0.0
f = 0.0
for i in range(len(anp)):
  e += abs(a[i] - b[i])
  f += (a[i]-b[i])**2.0
  pass

print(e,'\n', f)