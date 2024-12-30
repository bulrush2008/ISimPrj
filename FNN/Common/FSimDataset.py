
from torch.utils.data import Dataset
from pathlib import Path

import torch
import h5py

class FSimDataset(Dataset):
  def __init__(self, file:Path, caseList:list, varName:str):
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
    return self.numCases

  def __getitem__(self, idx):
    """
    return the input params and field
    """
    if idx >= self.numCases:
      raise IndexError(f"idx must be less than {self.numCases}")

    hdf = self.dataFile
    cid = self.caseList[idx]

    inp = hdf[cid]["InParam"][:]  # numpy.ndarray
    inp = torch.FloatTensor(inp)  # torch.FloatTensor

    data = []

    # 字典，三个键值："x","y","z"
    # 每个键后面，都连接这个二维列表数据，分别表示 8 个 block 的坐标数据
    #coords = {}

    # coords["x"] 将会包含所有的 block 的 x 坐标值，
    # coords["y"] 和 coords["z"] 同样如此
    coords = {"x":[], "y":[], "z":[]}
    #coords["x"] = []  # 2d list
    #coords["y"] = []  # ..
    #coords["z"] = []  # ..

    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-" + self.varName

      varFieldBlk = list(hdf[cid][key][:])
      data += varFieldBlk

      # coordx
      key = "Block-"+ "%02d"%blk + "-X"
      crd = list(hdf[cid][key][:])
      coords["x"] += crd

      # coordy
      key = "Block-"+ "%02d"%blk + "-Y"
      crd = list(hdf[cid][key][:])
      coords["y"] += crd

      # coordz
      key = "Block-"+ "%02d"%blk + "-Z"
      crd = list(hdf[cid][key][:])
      coords["z"] += crd
      pass

    return inp, torch.FloatTensor(data), coords
  
  def plotVTK(self, idx):
    pass
