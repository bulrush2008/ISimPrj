
import sys
import json

import h5py
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from Common.CaseSet import CaseSet
from Common.FSimDatasetPoint import FSimDatasetPoint
from Common.ModelPoint  import ModelPoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.MetricTracker import MetricTracker
from util.Write2HDF import write2HDF
from util.error import LInfError, L2Error, L1Error
import wandb


class FnnPoint(object):
#===============================================================================
  """
  - 应用任务类

  - 调用方法类和数据类，实现特定的应用任务
  """
  def __init__(self):
  #-----------------------------------------------------------------------------
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./fnnPoint.json", 'r') as inp:
      self.config = json.load(inp)

    self._setup_dataset()
    self._setup_model()
    pass

  def _setup_dataset(self):
  #-----------------------------------------------------------------------------
    self.train_batch_size = self.config["train_batch_size"]
    print(f"*Use train batch size: {self.train_batch_size}")
    case_set = CaseSet(ratio=self.config["test_ratio"])
    trn_set, tst_set = case_set.splitSet()
    self.trn_set = trn_set
    self.tst_set = tst_set
    self.file_path_h5 = Path(self.config["train_data"])
    self.fs_dataset_train = FSimDatasetPoint(file=self.file_path_h5, case_list=trn_set, var_name=self.config["var"])
    self.fs_dataset_test = FSimDatasetPoint(file=self.file_path_h5, case_list=tst_set, var_name=self.config["var"])
    self.train_loader = DataLoader(self.fs_dataset_train, batch_size=self.train_batch_size, shuffle=True, 
                                   num_workers=4, pin_memory=True)
    self.test_loader = DataLoader(self.fs_dataset_test, batch_size=self.train_batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True)
    pass

  def _setup_model(self):
  #-----------------------------------------------------------------------------
    self.model = ModelPoint(self.config)
    pass



  def train(self):
  #-----------------------------------------------------------------------------
    print(f"*Model Will Be Trained with Epochs {self.config['epochs']}.")

    self.test_LInf_tracker = {var: MetricTracker() for var in self.config["var"]}
    self.test_L2_tracker = {var: MetricTracker() for var in self.config["var"]}
    self.train_loss_tracker = MetricTracker()

    self.test_LInf_summary = {var: [] for var in self.config["var"]}
    self.test_L2_summary = {var: [] for var in self.config["var"]}
    self.train_loss_summary = []

    epochs = self.config["epochs"]

    for i in range(epochs):
      print(f" >> Training {self.config['var']}, epoch {i+1}/{epochs}")
      for params, coords, targets in tqdm(self.train_loader, desc=f"Epoch {i+1}/{epochs}"):
        batch_loss = self.model.train_step(params, coords, targets)
        self.train_loss_tracker.add(batch_loss)
        pass
      self.train_loss_summary.append(self.train_loss_tracker.summary())

      _, e_L2, e_LInf = self.evaluate(write_vtk=False)

      for i, var in enumerate(self.config["var"]):
        self.test_LInf_tracker[var].add(*[e_LInf_case[i] for e_LInf_case in e_LInf])
        self.test_L2_tracker[var].add(*[e_L2_case[i] for e_L2_case in e_L2])
        self.test_LInf_summary[var].append(self.test_LInf_tracker[var].summary())
        self.test_L2_summary[var].append(self.test_L2_tracker[var].summary())
      pass
    
      wandb_log = {"avg_train_loss": self.train_loss_tracker.average()}
      for var in self.config["var"]:
        wandb_log[f"avg_test_LInf_{var}"] = self.test_LInf_tracker[var].average()
        wandb_log[f"avg_test_L2_{var}"] = self.test_L2_tracker[var].average()
        self.test_LInf_tracker[var].reset()
        self.test_L2_tracker[var].reset()
      
      self.train_loss_tracker.reset()
      wandb.log(wandb_log)

    # directory of model
    dir_model = Path(self.config["dict_dir_path"])
    if not dir_model.exists(): dir_model.mkdir(parents=True)

    # save model parameters
    model_dicts_name = dir_model.joinpath(f"dict_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    torch.save(self.model.state_dict(), model_dicts_name)
    pass 
  
  def evaluate(self, write_vtk:bool=False):
    """
    """
    self.model.eval()
    self.fs_dataset_test.mode = "case"
    # error = {"L-inf": [], "L-2": [], "L-1": []}
    e_L1, e_L2, e_LInf = [], [], []

    for inp, axis_points, coord, value in tqdm(self.fs_dataset_test, desc="Evaluating"):
      with torch.no_grad():
        inp = inp.repeat(coord.shape[0], 1)
        pred = self.model.forward(params=inp, coords=coord)
        pred = pred.squeeze(1) # (1, P) # NOTE: This shit is weird but necessary
        if write_vtk:
          path = Path(self.config["eval_dir_path"]).joinpath(f"{inp[0][0]}_{inp[0][1]}_{inp[0][2]}.h5")
          write2HDF(pred, path, self.config["var"], axis_points)
        e_LInf.append(LInfError(pred, value)) # [len(var_name)]
        e_L2.append(L2Error(pred, value)) # [len(var_name)]
        e_L1.append(L1Error(pred, value)) # [len(var_name)]
        pass
      pass
    self.model.train()
    return e_L1, e_L2, e_LInf
      

  # def _train( self,
  #             dir_model:Path )->None:
  # #-----------------------------------------------------------------------------
  #   """
  #   Train the FnnPoint model by a give trainset, in which some cases field included.
  #   - dir_model: path of model
  #   """
  #   epochs = self.config["epochs"]

  #   for i in range(epochs):
  #     print(f" >> Training {self.config['var']}, epoch {i+1}/{epochs}")
  #     for params, coords, targets in tqdm(self.train_loader, desc=f"Epoch {i+1}/{epochs}"):
  #       batch_loss = self.model.train_step(params, coords, targets)
  #       self.train_loss_tracker.add(batch_loss)
  #       break
  #       pass
  #     # wandb.log({"avg_train_loss": self.train_loss_tracker.average()})
  #     self.train_loss_summary.append(self.train_loss_tracker.summary())
  #     self.train_loss_tracker.reset()

  #     test_error = self.evaluate(write_vtk=False)
  #     self.test_Linf_tracker.add(*test_error["L-inf"])
  #     self.test_L2_tracker.add(*test_error["L-2"])

      
  #     # wandb.log({"avg_test_Linf": self.test_Linf_tracker.average(), "avg_test_L2": self.test_L2_tracker.average()})
  #     self.test_Linf_summary.append(self.test_Linf_tracker.summary())
  #     self.test_L2_summary.append(self.test_L2_tracker.summary())
  #     self.test_Linf_tracker.reset()
  #     self.test_L2_tracker.reset()
  #     pass

  #   # save model parameters
  #   model_dicts_name = dir_model.joinpath(f"dict_{datetime.now().strftime('%Y%m%d%H%M%S')}")
  #   torch.save(self.model.state_dict(), model_dicts_name)
  #   pass  

  
if __name__ == "__main__":
  # Read wandb key from file
  try:
      with open("../wandb.key", 'r') as f:
          wandb_key = f.read().strip()
  except FileNotFoundError:
      raise FileNotFoundError("wandb.key file not found. Please create a file named 'wandb.key' containing your wandb API key.")

  wandb.login(key=wandb_key)
  wandb.init(project="pinn_08_07")
  fnn_point = FnnPoint()
  fnn_point.train()
  # e = fnn_point.evaluate(write_vtk=True)
  # print(e)