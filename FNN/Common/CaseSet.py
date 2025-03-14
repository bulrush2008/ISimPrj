
import numpy as np

class CaseSet(object):
#===============================================================================
  def __init__(self, ratio=0.2):
  #-----------------------------------------------------------------------------
    # all cases list
    self.idxList = list(range(1,126)) # [1,2,3,..., 124,125]
    
    self.size = len(self.idxList)
    self.ratio = ratio

    # info of each case
    self.blockNumb = 8

    self.blockInfo = []
    self.blockPosi = []
    
    self._calcBlockInfo()

  def __len__( self ):
  #-----------------------------------------------------------------------------
    """
    Now len[obj] is ok
    """
    return self.size

  def __getitem__(self, idx):
  #-----------------------------------------------------------------------------
    """
    so you use the class as: obj[idx]
    """
    if idx >= self.size:
      raise IndexError(f"{idx} beyond range.")
    return self.idxList[idx]

  def _calcBlockInfo(self):
  #-----------------------------------------------------------------------------
    calcPtsNum = lambda l: (l[1]-l[0]+1) * (l[3]-l[2]+1) * (l[5]-l[4]+1)

    # for block 0
    ptsB0 = [2,27,2,52,2,12]
    numbPtsB0 = calcPtsNum(ptsB0)

    self.blockInfo.append(ptsB0)
    self.blockPosi.append(numbPtsB0 + 0)

    # for block 1
    ptsB1 = [2,27,2,52,2,13]
    numbPtsB1 = calcPtsNum(ptsB1)

    self.blockInfo.append(ptsB1)
    self.blockPosi.append(numbPtsB1 + self.blockPosi[0])

    # for block 2
    ptsB2 = [2,27,2,53,2,12]
    numbPtsB2 = calcPtsNum(ptsB2)

    self.blockInfo.append(ptsB2)
    self.blockPosi.append(numbPtsB2 + self.blockPosi[1])

    # for block 3
    ptsB3 = [2,27,2,53,2,13]
    numbPtsB3 = calcPtsNum(ptsB3)

    self.blockInfo.append(ptsB3)
    self.blockPosi.append(numbPtsB3 + self.blockPosi[2])

    # for block 4
    ptsB4 = [2,28,2,52,2,12]
    numbPtsB4 = calcPtsNum(ptsB4)

    self.blockInfo.append(ptsB4)
    self.blockPosi.append(numbPtsB4 + self.blockPosi[3])

    # for block 5
    ptsB5 = [2,28,2,52,2,13]
    numbPtsB5 = calcPtsNum(ptsB5)

    self.blockInfo.append(ptsB5)
    self.blockPosi.append(numbPtsB5 + self.blockPosi[4])

    # for block 6
    ptsB6 = [2,28,2,53,2,12]
    numbPtsB6 = calcPtsNum(ptsB6)

    self.blockInfo.append(ptsB6)
    self.blockPosi.append(numbPtsB6 + self.blockPosi[5])

    # for block 7
    ptsB7 = [2,28,2,53,2,13]
    numbPtsB7 = calcPtsNum(ptsB7)

    self.blockInfo.append(ptsB7)
    self.blockPosi.append(numbPtsB7 + self.blockPosi[6])
    pass

  def splitSet(self):
  #-----------------------------------------------------------------------------
    """
    - 用于测试本文件内的类或函数

    - Split the caseSet into train and test sets
    """
    numbOfTrnSet = np.int64( self.size * (1.0-self.ratio) )

    # 1984 is a seed
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

    # type: list of strings
    return trnSet, tstSet

if __name__=="__main__":

  cases = CaseSet( ratio=0.2 )

  listTrn, listTst = cases.splitSet()
  print(listTrn, "\n", listTst)

  #print(cases.blockNumb)
  #for iblk in range(cases.blockNumb):
  #  print(cases.blockInfo[iblk])
  #  print(cases.blockPosi[iblk])