
from torch.utils.data import Dataset
from pathlib import Path

import torch
import numpy as np
import h5py

class FSimDatasetPoint(Dataset):
#===============================================================================
  """
  - FSimDatasetPoint: Dataset for Point training on FSim data
  """
  def __init__(self, file:Path, caseList:list, varName:str, mode:str="serial"):
  #-----------------------------------------------------------------------------
    """
    """
    # self.dataFile = h5py.File(file, 'r')
    self._mode = mode
    self.caseList = caseList
    self.numCases = len(caseList)

    if varName not in ["P", "U", "V", "W", "T"]:
      raise ValueError("Error: the Variable Name must be P/U/V/W/T.")

    self.varName  = varName

    self.inp_serial   = []
    self.coord_serial   = []
    self.value_serial   = []

    self.inp_case = []
    self.axis_points_case = []
    self.coord_case = []
    self.value_case = []

    with h5py.File(file, 'r') as dataFile:
      for case in self.caseList:
        inp = dataFile[case]["input"]["inp"][:].astype(np.float32)
        # print(f"case: {case}, inp: {inp}")
        axis_points = {"x":[], "y":[], "z":[]}
        values = np.empty((0, 1), dtype=np.float32)
        coords = np.empty((0, 3), dtype=np.float32)
        for blk in range(8):
          key = f"block{blk:03d}"

          # Load coordinates
          x = dataFile[case][key]["X"][:]
          y = dataFile[case][key]["Y"][:]
          z = dataFile[case][key]["Z"][:]

          axis_points["x"] += list(x)
          axis_points["y"] += list(y)
          axis_points["z"] += list(z)

          Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # shape: (x,y,z)

          # Flatten to (N, 3)
          coords_stack = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
          coords = np.concatenate([coords, coords_stack], axis=0)

          # Flatten var field
          T = dataFile[case][key][self.varName][:].astype(np.float32).reshape(-1, 1)
          values = np.concatenate([values, T], axis=0)

          # Repeat `inp` for each point in block
          inp_repeated = np.repeat(inp[np.newaxis, :], coords_stack.shape[0], axis=0)

          self.inp_serial.append(inp_repeated)
          self.coord_serial.append(coords_stack)
          self.value_serial.append(T)

        self.inp_case.append(torch.tensor(inp, dtype=torch.float32))
        self.axis_points_case.append(axis_points)
        self.coord_case.append(torch.tensor(coords, dtype=torch.float32))
        self.value_case.append(torch.tensor(values, dtype=torch.float32).T)
    
    self.inp_serial  = np.concatenate(self.inp_serial, axis=0)
    self.coord_serial  = np.concatenate(self.coord_serial, axis=0)
    self.value_serial  = np.concatenate(self.value_serial, axis=0)
    assert self.inp_serial.shape[0] == self.coord_serial.shape[0] == self.value_serial.shape[0]
    
  def __len__(self):
    if self.mode == "serial":
      return self.inp_serial.shape[0]
    elif self.mode == "case":
      return len(self.inp_case)
    else:
      raise ValueError(f"Invalid mode: {self.mode}")

  def __getitem__(self, idx):
    if self.mode == "serial":
      return (
        torch.from_numpy(self.inp_serial[idx]).to(torch.float32), #设置为float32，由于MPS不支持float64
        torch.from_numpy(self.coord_serial[idx]).to(torch.float32),
        torch.from_numpy(self.value_serial[idx]).to(torch.float32)
      )
    elif self.mode == "case":
      return (
        self.inp_case[idx],
        self.axis_points_case[idx],
        self.coord_case[idx],
        self.value_case[idx]
      )
    else:
      raise ValueError(f"Invalid mode: {self.mode}")
  
  @property
  def mode(self):
    return self._mode
  
  @mode.setter
  def mode(self, val):
    if val not in ["serial", "case"]:
      raise ValueError(f"Invalid mode: {val}")
    self._mode = val
    
  
if __name__ == "__main__":
  from CaseSet import CaseSet
  caseSet = CaseSet(ratio = 0.2)
  trnSet, tstSet = caseSet.splitSet()
  import time 
  start_time = time.time()
  fsDataset = FSimDatasetPoint(Path("../FSCases/FSHDF/MatrixData.h5"), trnSet, "T")
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

