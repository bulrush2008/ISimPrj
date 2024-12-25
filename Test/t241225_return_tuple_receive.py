
def func(i, j):
  return i, j, i+j

i = 4; j = 5
#x,y,z = func(i,j)  # correct maner
#(x,y,z) = func(i,j) # correct maner 2
(x,
 y,
 z) = func(i,j) # correct maner 3
print(x,y,z)