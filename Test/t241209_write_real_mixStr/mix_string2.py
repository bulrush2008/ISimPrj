
n = 2
#x = "%03d"%n.encode("utf-8") # wrong, because <"%d"%n is int, not str>
#x = str("%03d"%n).encode("utf-8")  # yes, 0 padding
#x = str("%4d"%n).encode("utf-8")  # yes, no padding
x = ("%5s"%n).encode("utf-8")
print(x, type(x))

y = b"what is your score: " + x
print(y)

vtk = open("t01.vtk", 'wb')
vtk.write(y)

vtk.close()
