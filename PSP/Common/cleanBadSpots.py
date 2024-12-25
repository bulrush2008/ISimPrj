

import numpy as np

def cleanBadSpots(field:np.ndarray, gIndexRange:list)->None:
  """
  read the data from original vtr file, clean the bad spots, e.g. zero points
  at the vertexes

  - field: any flow field, including T, P, U/V/W, of one block. The format is
    of type numpy.ndarray
  """

  ni = gIndexRange[0][1] - gIndexRange[0][0] + 1  
  nj = gIndexRange[1][1] - gIndexRange[1][0] + 1  
  nk = gIndexRange[2][1] - gIndexRange[2][0] + 1

  #print(gIndexRange)
  #print(ni,nj,nk)
  field = np.reshape(field, (ni,nj,nk))

  # [0,ni-1], [0,nj-1], [0,nk-1]
  field[0,0,0] = field[1,1,1]
  field[0,0,nk-1] = field[1,1,nk-2]

  field[ni-1,0,0] = field[ni-2, 1,1]
  field[ni-1,0,nk-1] = field[ni-2, 1, nk-2]

  field[ni-1,nj-1,0] = field[ni-2,nj-2,1]
  field[ni-1,nj-1,nk-1] = field[ni-2,nj-2,nk-2]

  field[0,nj-1,0] = field[1,nj-2,1]
  field[0,nj-1,nk-1] = field[1,nj-2,nk-2]

  pass

if __name__=="__main__":
  gir = [[2,5],[3,7],[4,9]]

  f = np.zeros(120)
  f = np.reshape(f, (4,5,6))

  for i in range(4):
    for j in range(5):
      for k in range(6):
        f[i,j,k] = i*100 + j*10 + k
  f = np.reshape(f, (120,))
  print(f)

  #cleanBadSpots(f, gir)