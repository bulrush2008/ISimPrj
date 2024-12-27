
a = "Hello World"
print(a[0])


b = b"Hello World"
print(b[0])
print(b)
print(len(a), len(b))

for i in range(len(a)):
  print(a[i], b[i])

c = a.encode("ASCII")
print(c)

if b == c:
  print("b == c")