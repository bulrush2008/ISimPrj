
import numpy as np

class CaseSet:
  def __init__(self, ratio=0.2):
    self.idxList = [1,   3,   5,   8,   11,  13,  15,
                    18,  21,  23,  25,  28,  31,  34,
                    37,  40,  43,  46,  49,  51,  53,
                    55,  58,  61,  63,  65,  68,  71,
                    73,  75,  78,  81,  84,  87,  90,
                    93,  96,  99,  101, 103, 105, 108,
                    111, 113, 115, 118, 121, 123, 125]
    
    self.size = len(self.idxList)
    self.ratio = ratio

  def __len__( self ):
    """
    Now len[obj] is ok
    """
    return self.size

  def __getitem__(self, idx):
    """
    so you use the class as: obj[idx]
    """
    if idx >= self.size:
      raise IndexError(f"{idx} beyond range.")
    return self.idxList[idx]

  def splitSet(self):
    """
    Split the caseSet into train and test sets
    """
    numbOfTrnSet = np.int64( self.size * (1.0-self.ratio) )

    # 1984 is a seed, other ints also OK
    np.random.seed(1984)
    permut = np.random.permutation( self.size )

    # give the case names list in train set
    trnSet = []
    for i in permut[:numbOfTrnSet]:
      theCase = "C" + "%03d"%(self.idxList[i])
      trnSet.append(theCase)

    # give the case names list in test set
    tstSet = []
    for i in permut[numbOfTrnSet:]:
      theCase = "C" + "%03d"%(self.idxList[i])
      tstSet.append(theCase)

    return trnSet, tstSet

if __name__=="__main__":

  cases = CaseSet( ratio=0.2 )

  print(len(cases), cases.size)
  print(cases[4], cases[43]) # 11, 113

  listTrn, listTst = cases.splitSet()
  print(listTrn, "\n", listTst)