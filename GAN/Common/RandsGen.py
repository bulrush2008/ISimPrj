
import torch
import numpy as np
import random

from Common.paramInps import ParametersInput as PInp

class RandsGen(object):
#===============================================================================
  def __init__(self, seed:int=42):
  #-----------------------------------------------------------------------------
    random.seed(seed)  # 标准库随机数生成包
    self.pinp = PInp()

    rand_idx = random.randint(0,99)

    rand_key = self.pinp.trnCaseList[rand_idx]
    a_temp = self.pinp.prmDict[rand_key]
    self.inpu = torch.FloatTensor(a_temp) # single precision
    #...........................................................................

    np.random.seed(seed=seed) # 第三方库
    b_temp = np.random.randn(100)
    self.seed = torch.from_numpy(b_temp)#.double()
    pass

  def update_inpu(self)->None:
  #-----------------------------------------------------------------------------
    """
    random array of size 3, which equals that of parameterized inputs
    """
    rand_idx = random.randint(0,99)

    rand_key = self.pinp.trnCaseList[rand_idx]
    a_temp = self.pinp.prmDict[rand_key]
    self.inpu = torch.FloatTensor(a_temp) # single precision
    pass

  def update_seed(self)->None:
  #-----------------------------------------------------------------------------
    """
    normalized Gaussian districution
    """
    b_temp = np.random.randn(100)
    self.seed = torch.from_numpy(b_temp)#.double()
    pass
  pass  # end class "RandsGen"

if __name__=="__main__":
#===============================================================================
  rg = RandsGen(42)

  for i in range(5):
    a = rg.inpu; print(f"input = {a}")
    rg.update_inpu()

    #b = rg.seed; print(f"seed[32]  = {b[32]}")
    #rg.update_seed()
    pass
  pass