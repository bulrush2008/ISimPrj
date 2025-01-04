
import pandas as pd
import openpyxl

# 读取 Excel 文件
excel_file = "ROM-Cases-Records.xlsx"

df = pd.read_excel(excel_file)

#df.info()

#print(type(df.iloc[20:145,1:4]))
l = df.iloc[20:145,1:4].values.tolist()

length = len(l); print(length)

print(l)

# 将数据保存为 CSV 文件
#csv_file = 'data.csv'
#df.to_csv(csv_file, index=False)

#print(f"Excel 文件已成功转换为 {csv_file}")