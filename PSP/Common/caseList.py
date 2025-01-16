"""
read list of case names from 'PSP/PSP.inp'
"""
import json
from pathlib import Path

def PSP_read_json(jsonFile:Path):
  with open(jsonFile, 'r') as inp:
    data = json.load(inp)

  caseList = data["case"]
  return caseList

if __name__=="__main__":
  jsonFile = Path("../PSP.json")
  caseList = PSP_read_json(jsonFile=jsonFile)

  print(len(caseList))

  for case in caseList:
    print(case)