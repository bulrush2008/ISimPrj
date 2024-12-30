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

def splitData(idxBlk:int, infoBlk:list)->tuple:
  f = lambda l: (l[1]-l[0]+1, l[3]-l[2]+1, l[5]-l[4]+1)
  g = lambda t: t[0]*t[1]*t[2]

  #
  pass