
import h5py
import numpy as np

groups = ["train", "test", "model"]

fileListTrain = ["Data01.dat", "Data02.dat", "Data03.dat"]
fileListTest  = ["Data04.dat", "Data05.dat"]
fileListModel = ["Data06.dat"]

#def Write2HDF(inData:np.ndarray, HDF5FileName, groupName, datasetName):
#  with h5py.File(HDF5FileName, 'a') as hdf:
#    if groupName in hdf.:
#      group = hdf[groupName]
#      group.create_dataset(datasetName, data=inData)
#    else:
#      group = hdf.create_group(groupName)
#      group.create_dataset(datasetName, data=inData)

    #print(hdf[groupName][datasetName])
#    pass
#  pass

#inData = np.loadtxt(fileListTest[0], delimiter=',')
#Write2HDF(inData, "newMultiGroups.h5", groups[0], fileListTest[0])

with h5py.File("multiGroups.h5", 'w') as hdf:
  for id, grpname in enumerate(groups):
    igroup = hdf.create_group(grpname)

    if id==0:
      for id2, file in enumerate(fileListTrain):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)

#        #print(hdf["train"]["Data01.dat"])
    elif id==1:
      for id2, file in enumerate(fileListTest):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)
    elif id==2:
      for id2, file in enumerate(fileListModel):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)

  for key in hdf.keys():
    print(hdf[key].name)
    