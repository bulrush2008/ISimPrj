
import csv

# 初始化一个空字典来存储数据
data_dict = {}

# 打开 CSV 文件
with open('./inp.csv', mode='r', encoding='utf-8') as csv_file:
    # 创建 CSV 读取器
    csv_reader = csv.reader(csv_file)

    next(csv_reader)
    
    # 遍历每一行
    for row in csv_reader:
        # 第一列作为键，第二列作为值
        key = row[0]
        value = row[1]
        
        # 尝试将值转换为浮点数或整数
        #try:
        #    value = float(value)
        #    if value.is_integer():
        #        value = int(value)
        #except ValueError:
        #    # 如果转换失败，保持为字符串
        #    pass
        
        # 将键值对存入字典
        data_dict[key] = value

# 打印字典以验证结果

for key, value in data_dict.items():
  print(key, type(key), value, type(value))