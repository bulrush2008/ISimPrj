
import csv

inp_file = open("inp.csv", mode='r', encoding="utf-8")

csv_reader = csv.reader(inp_file)

for row in csv_reader:
  print(row, type(row), type(row[0]))