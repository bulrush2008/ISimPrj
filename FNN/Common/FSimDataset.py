
from torch.utils.data import Dataset
from pathlib import Path

import torch
import h5py

class FSimDataset(Dataset):
  def __init__(self, file:Path, caseList:list):
    """
    - The data remain stored in h5 file, read when needed

    - inputs:
    /caseList/: list of cases names, made of set either "test" or "train"

    - member data: self.x
      - dataFile: HDF5 file
      - caseList: dataList, a char string list in hdf5 file
      - numCases: number of cases input
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList
    self.numCases = len(caseList)

  def __len__(self):
    return self.numCases

  def __getitem__(self, idx):
    """
    return the input params and field
    """
    if idx >= self.numCases:
      raise IndexError

    hdf = self.dataFile
    cid = self.caseList[idx]

    inp = hdf[cid]["InParam"][:]
    inp = torch.FloatTensor(inp)

    data = []
    coords = {}

    coords["x"] = [[]]
    coords["y"] = [[]]
    coords["z"] = [[]]

    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-P"

      presFieldBlk = list(hdf[cid][key][:])
      data += presFieldBlk

      # coordx
      key = "Block-"+ "%02d"%blk + "-X"
      crd = list(hdf[cid][key][:])
      coords["x"].append(crd)

      # coordy
      key = "Block-"+ "%02d"%blk + "-Y"
      crd = list(hdf[cid][key][:])
      coords["y"].append(crd)

      # coordz
      key = "Block-"+ "%02d"%blk + "-Z"
      crd = list(hdf[cid][key][:])
      coords["z"].append(crd)
      pass

    del coords["x"][0]
    del coords["y"][0]
    del coords["z"][0]

    return inp, torch.FloatTensor(data), coords
  
  def plotVTK(self, idx):
    pass
