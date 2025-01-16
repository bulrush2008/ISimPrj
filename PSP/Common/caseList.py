"""
read list of case names from 'PSP/PSP.inp'
"""
import csv
from pathlib import Path

def PSP_read_csv(csvfile:Path):
  inp = open(csvfile, mode='r', encoding="utf-8")
  csvObj = csv.reader(inp)

  caseList = []
  for row in csvObj:
    caseList.append(row[0])

  #numOfCases = len(caseList)
  return caseList

if __name__=="__main__":
  csvfile = Path("../PSP.inp")
  caseList = PSP_read_csv(csvfile=csvfile)

  print(len(caseList))

  for case in caseList:
    print(case)