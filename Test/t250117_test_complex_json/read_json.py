
import json

with open("./inp.json", 'r') as inp:
  data = json.load(inp)
  pass

print(data["vars"])
print(type(data["vars"]))