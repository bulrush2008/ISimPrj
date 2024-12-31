"""
Split data from the total in h5 file, for a certain block.

    numCoordsEachBlk = [[2,27,2,52,2,12],
                        [2,27,2,52,2,13],
                        [2,27,2,53,2,12],
                        [2,27,2,53,2,13],
                        [2,28,2,52,2,12],
                        [2,28,2,52,2,13],
                        [2,28,2,53,2,12],
                        [2,28,2,53,2,13]]
"""

def splitData(idxBlk:int, infoBlk:list)->list:
  """
  idxBlk: block index
  infoBlk: information including all block's x/y/z dims

  return: a position list, including data array index positions of var, X, Y, Z
  """
  f = lambda l: (l[1]-l[0]+1, l[3]-l[2]+1, l[5]-l[4]+1)
  g = lambda t: t[0]*t[1]*t[2]

  posi = {"Var":[], "X":[], "Y":[], "Z":[]}

  numBlks = len(infoBlk)  # 8 blocks
  print(numBlks)

  vf = xf = yf = zf = 0

  for iblk in range(numBlks):

    x, y, z = f(infoBlk[iblk])

    xf += x
    yf += y
    zf += z

    posi["X"].append(xf)
    posi["Y"].append(yf)
    posi["Z"].append(zf)

    vf += x*y*z

    posi["Var"].append(xf*yf*zf)
    pass

  return posi

if __name__=="__main__":
  numCoordsEachBlk = [[2,27,2,52,2,12],
                      [2,27,2,52,2,13],
                      [2,27,2,53,2,12],
                      [2,27,2,53,2,13],
                      [2,28,2,52,2,12],
                      [2,28,2,52,2,13],
                      [2,28,2,53,2,12],
                      [2,28,2,53,2,13]]

  p = splitData(1, numCoordsEachBlk)

  print(p)

  pass