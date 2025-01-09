
import csv

def read_csv(file:str):
# 初始化一个空字典来存储数据
  data_dict = {}

  # 打开 CSV 文件
  with open(file, mode='r', encoding='utf-8') as csv_file:
    # 创建 CSV 读取器
    csv_reader = csv.reader(csv_file)

    # 第一行乃文件说明信息，略去
    next(csv_reader)
    
    # 遍历每一行
    for row in csv_reader:
      # 第一列作为键，第二列作为值
      key = row[0]
      value = row[1]

      # 将键值对存入字典
      data_dict[key] = value
      pass
    pass

  return data_dict

# this the main function of FSimPrj

file = r"./fsim.inp"
data_csv = read_csv(file)

print(data_csv)