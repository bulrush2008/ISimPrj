
"""
This is function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      2025.3.27   Simpop.cn   v4.x
"""
import sys
import json

import h5py
import torch

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Generator import Generator
from Common.Discriminator import Discriminator

from Common.RandsGen import RandsGen

class GAN_Train(object):
#===============================================================================
  def __init__( self ):
  #-----------------------------------------------------------------------------
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./GAN_Train.json", 'r') as inp:
      data = json.load(inp)
      pass

    ratioTest = data["test_ratio"]  # eg 0.2
    caseSet = CaseSet( ratio=ratioTest )

    trnSet, tstSet = caseSet.splitSet()

    self.trnSet = trnSet
    self.tstSet = tstSet

    # path of data used as training and possibly test
    self.filePathH5 = Path(data["train_data"])

    self.fieldList = data["vars"]

    self.rand_generator = RandsGen(1984)
    pass

  def train( self ):
  #-----------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter
    #fieldList = {"T":1, "P":1}
    fieldList = self.fieldList

    print(f"*Fields Models Will Be Trained with Epochs {fieldList}.")

    trnSet = self.trnSet
    tstSet = self.tstSet

    filePathH5 = self.filePathH5

    models = self._train( varList  = fieldList,
                          trainSet = trnSet,
                          testSet  = tstSet,
                          dataPath = filePathH5 )

    dirPNG = Path("./Pics")
    if not dirPNG.exists(): dirPNG.mkdir(parents=True)

    dirModel = Path("./StateDicts")
    if not dirModel.exists(): dirModel.mkdir(parents=True)

    for var in fieldList.keys():
      #------------------------------------------------------------------------
      # G & D: plot loss history and save
      models[var][0].saveLossHistory2PNG(dirPNG)  # G
      models[var][1].saveLossHistory2PNG(dirPNG)  # D

      #------------------------------------------------------------------------
      # save model parameters
      # G
      model_dicts_name = dirModel.joinpath(f"G_Dict_{var}.pth")
      torch.save(models[var][0].model.state_dict(), model_dicts_name)
      # D
      model_dicts_name = dirModel.joinpath(f"D_Dict_{var}.pth")
      torch.save(models[var][1].model.state_dict(), model_dicts_name)
      pass
    pass

  def _train( self,
              varList :dict,
              trainSet:list,
              testSet :list,
              dataPath:Path )->dict:
  #-----------------------------------------------------------------------------
    """
    Train the GAN model by a give trainset, in which some cases field included.
    - varList : dict of epochs for each field, such as ["P":1,"T":2]
    - trainSet: list of case names in train set, each is a string
    - testSet : list of case names in test set, each is a string
    - dataPath: path of data of train set
    """
    # extract the var names
    fields = []
    for key in varList.keys():
      fields.append(key)
      pass
  
    # including all trained models
    models = {}

    # train fields
    for var in fields:
      # obj to get the train data set
      fsDataset_train = FSimDataset(dataPath, trainSet, var)

      # gen a obj as regression, and then train the model
      varG_dict_path = Path(f"./StateDicts/G_Dict_{var}.pth")
      varD_dict_path = Path(f"./StateDicts/D_Dict_{var}.pth")

      if varG_dict_path.exists() and varD_dict_path.exists():
        print(f"Train from G_Dict_{var}.pth & D_Dict_{var}.pth")
      else:
        varG_dict_path = None
        varD_dict_path = None
        print(f"Train from ZERO for {var}")
        pass

      G = Generator(var, varG_dict_path)
      D = Discriminator(var, varD_dict_path)

      print(f"*GAN: Now we are training {var:>3}:")

      # train the model
      epochs = varList[var]

      for i in range(epochs):
        print(f" >> Training {var}, epoch {i+1}/{epochs}")
        for inp, fld, _ in fsDataset_train:
          # train discriminatro on True
          D.train(fld, inp, torch.FloatTensor([1.0]))

          # 为鉴别器生成随机种子和标签
          label_inp = self.rand_generator.inpu
          seeds_inp = self.rand_generator.seed

          # train discriminator on False
          D.train(G.forward(seeds_inp, label_inp).detach(), label_inp, torch.FloatTensor([0.0]))

          # train generator
          G.train(D, seeds_inp, label_inp, torch.FloatTensor([1.0]))

          self.rand_generator.update_seed()
          self.rand_generator.update_inpu()
          pass  # Tranverse all fields in 1 epoch
        pass  # All Epochs Finished

      models[var] = [G,D]
      pass

    # now all variable models have been trained
    return models
  pass  # end class

if __name__=="__main__":
#===============================================================================
  print("Only define class 'GAN_Train' here")
  pass