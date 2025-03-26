
import torch
import numpy as np

class RandsGen(object):
#===============================================================================
  def __init__(self, seed:int=42):
  #-----------------------------------------------------------------------------
    np.random.seed(seed=seed)

    a_temp = np.random.rand(3)
    self.inpu = torch.from_numpy(a_temp).double()

    b_temp = np.random.randn(100)
    self.seed = torch.from_numpy(b_temp).double()
    pass

  def update_inpu(self)->None:
  #-----------------------------------------------------------------------------
    """
    random array of size 3, which equals that of parameterized inputs
    """
    a_temp = np.random.rand(3)
    self.inpu = torch.from_numpy(a_temp).double()
    pass


  def update_seed(self)->None:
  #-----------------------------------------------------------------------------
    """
    normalized Gaussian districution
    """
    b_temp = np.random.randn(100)
    self.seed = torch.from_numpy(b_temp).double()
    pass
  pass  # end class "RandsGen"

if __name__=="__main__":
#===============================================================================
  rg = RandsGen(42)

  for i in range(5):
    #a = rg.inpu; print(f"input = {a}")
    #rg.update_inpu()

    b = rg.seed; print(f"seed[32]  = {b[32]}")
    rg.update_seed()
  pass