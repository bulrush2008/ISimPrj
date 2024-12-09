
#import io

#f = io.BytesIO(b"What is the number? ")
#print(f.getvalue())

x = str(2).encode("utf-8")
print(x, type(x))

y = b"what is your score: " + x
print(y)

vtk = open("t01.vtk", 'wb')
vtk.write(y)

vtk.close()
