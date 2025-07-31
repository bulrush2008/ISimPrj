
from torch.utils.data import Dataset
from pathlib import Path

import torch
import numpy as np
import h5py

class FSimDatasetPINN(Dataset):
#===============================================================================
  """
  - FSimDatasetPINN: Dataset for PINN training on FSim data
  """
  def __init__(self, file:Path, caseList:list, varName:str):
  #-----------------------------------------------------------------------------
    """
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList
    self.numCases = len(caseList)

    if varName not in ["P", "U", "V", "W", "T"]:
      raise ValueError("Error: the Variable Name must be P/U/V/W/T.")

    self.varName  = varName
    self.data = []
    self.inputs   = []
    self.coords   = []
    self.values   = []

    for case in self.caseList:
      inp = self.dataFile[case]["input"]["inp"][:].astype(np.float32)
      for blk in range(8):
        key = f"block{blk:03d}"

        # Load coordinates
        x = self.dataFile[case][key]["X"][:]
        y = self.dataFile[case][key]["Y"][:]
        z = self.dataFile[case][key]["Z"][:]
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # shape: (x,y,z)

        # Flatten to (N, 3)
        coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Flatten var field
        T = self.dataFile[case][key][self.varName][:].astype(np.float32).reshape(-1, 1)

        # Repeat `inp` for each point in block
        inp_repeated = np.repeat(inp[np.newaxis, :], coords.shape[0], axis=0)

        self.inputs.append(inp_repeated)
        self.coords.append(coords)
        self.values.append(T)
    
    self.inputs  = np.concatenate(self.inputs, axis=0)
    self.coords  = np.concatenate(self.coords, axis=0)
    self.values  = np.concatenate(self.values, axis=0)
    assert self.inputs.shape[0] == self.coords.shape[0] == self.values.shape[0]
    
  def __len__(self):
    return self.inputs.shape[0]


  def __getitem__(self, idx):
    return (
      torch.from_numpy(self.inputs[idx]).to(torch.float32), #设置为float32，由于MPS不支持float64
      torch.from_numpy(self.coords[idx]).to(torch.float32),
      torch.from_numpy(self.values[idx]).to(torch.float32)
    )

if __name__ == "__main__":
  from CaseSet import CaseSet
  caseSet = CaseSet(ratio = 0.2)
  trnSet, tstSet = caseSet.splitSet()
  import time 
  start_time = time.time()
  fsDataset = FSimDatasetPINN(Path("../FSCases/FSHDF/MatrixData.h5"), trnSet, "T")
  print(len(fsDataset))
  end_time = time.time()
  print(f"Time taken: {end_time - start_time} seconds")
  from torch.utils.data import DataLoader
  train_loader = DataLoader(fsDataset, batch_size=3, shuffle=True)
  for batch in train_loader:
    inputs, coords, values = batch
    print(f"Inputs shape: {inputs.shape}, Coords shape: {coords.shape}, Values shape: {values.shape}")
    print("First input:", inputs[0])
    print("First coords:", coords[0])
    print("First value:", values[0])

