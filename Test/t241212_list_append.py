
a = [1,2,3,4]
print(a)

#b = []
a.append(3.5)
print(a)

b = [[1,2,3]]
print(b)
print(b[0])

b.append([2,3,5])
print(b)
print(type(b), len(b))

c = [[]]
print(len(c))
print(c)

c.append([1,2])
print(len(c))
print(c)

del c[0]
print(len(c))
print(c)




