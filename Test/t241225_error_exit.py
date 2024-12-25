
s = input("Input 'Y' or 'N':\n")

if s == 'Y':
  raise LookupError("Y")
else:
  raise LookupError("N")

print("Can this line printed in screen?")