
'''
import numpy as np

with open("./demo.dat", "rb") as f:
  f.seek(22)
  data_ints = np.fromfile(f, dtype=np.int32, count=1)
  print(data_ints)

  data_floats = np.fromfile(f, dtype=np.float64, count=data_ints[0])
  print(data_floats)
  print(len(data_floats))
'''

import numpy as np

offset = 0

with open("./demo.dat", "rb") as f:
  line = f.readline(); offset += len(line)#; print(offset)

  char = f.read(1).decode("ASCII")
  print(char, "line 9")
  #f.seek(offset+1) # seek is not necessary

  data_ints = np.fromfile(f, dtype=np.int32, count=1)
  print(data_ints)

  data_floats = np.fromfile(f, dtype=np.float64, count=data_ints[0])
  print(data_floats)
  print(len(data_floats))