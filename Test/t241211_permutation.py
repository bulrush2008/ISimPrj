
import numpy as np

numbOfData = 27
ratio = 0.2

sizeOfTest = np.int32(numbOfData * ratio)
print("Size of test set: ", sizeOfTest)

a = np.random.permutation(numbOfData)

listTest  = a[:sizeOfTest]
listTrain = a[sizeOfTest:]

print(len(a));         print(a)
print(len(listTest));  print(listTest)
print(len(listTrain)); print(listTrain)

print(type(a))

