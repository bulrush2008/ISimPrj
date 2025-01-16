
import json

with open("inp.json", 'r') as f:
  data = json.load(f)

nx = data["grid"]["nx"]

print(f"nx = {nx}")

vel = data["vel"]
print(vel, type(vel))