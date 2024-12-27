
import h5py

with h5py.File("./combined_data.h5", 'r') as hdf:
  for group in hdf:
    print(group)
    print(type(group))

  #fileList = hdf.keys()
  #print(type(fileList), fileList)