
#from torch.utils.data import Dataset
from pathlib import Path

import torch
import h5py

#class FSimDataset(Dataset):
class FSimDataset(object):
#===============================================================================
  """
  - 数据类

  - 打开 H5 矩阵数据库，供随时调用
  - 此类本身并不存储数据，而是提供一个指向数据的地址
  """
  def __init__(self, file:Path, caseList:list, varName:str):
  #-----------------------------------------------------------------------------
    """
    - The data remain stored in h5 file, read when needed

    - inputs:
    /caseList/: list of cases names, made of set either "test" or "train"

    - member data: self.x
      - dataFile: HDF5 file
      - caseList: dataList, a char string list in hdf5 file
      - numCases: number of cases input
      - varName : "P/U/V/W/T", other is for now illegal
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList
    self.numCases = len(caseList)

    if varName not in ["P", "U", "V", "W", "T"]:
      raise ValueError("Error: the Variable Name must be P/U/V/W/T.")

    self.varName  = varName

  def __len__(self):
  #-----------------------------------------------------------------------------
    return self.numCases

  def __getitem__(self, idx):
  #-----------------------------------------------------------------------------
    """
    return the input params and field
    """
    if idx >= self.numCases:
      raise IndexError(f"idx must be less than {self.numCases}")

    hdf = self.dataFile
    cid = self.caseList[idx]

    inp = hdf[cid]["input"]["inp"][:]  # numpy.ndarray
    inp = torch.FloatTensor(inp)  # convert to type torch.FloatTensor

    data = []

    # coords["x"] 将会包含所有的 block 的 x 坐标值，
    # coords["y"] 和 coords["z"] 同样如此
    coords = {"x":[], "y":[], "z":[]}

    for blk in range(8):
      #key = "Block-"+ "%02d"%blk + "-" + self.varName
      key = f"block{blk:03d}"

      # for the variable field
      varFieldBlk = list(hdf[cid][key][self.varName][:])
      data += varFieldBlk

      # coord x
      crd = list(hdf[cid][key]["X"][:])
      coords["x"] += crd

      # coord y
      crd = list(hdf[cid][key]["Y"][:])
      coords["y"] += crd

      # coord z
      crd = list(hdf[cid][key]["Z"][:])
      coords["z"] += crd
      pass

    return inp, torch.FloatTensor(data), coords

  def plotVTK(self, idx):
  #-----------------------------------------------------------------------------
    pass  # end class func plotVTK
  pass  # end class FSimDataset