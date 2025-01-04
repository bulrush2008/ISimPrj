
import pandas as pd
import openpyxl

# 读取 Excel 文件
excel_file = "ROM工况（包含第三次提交）.xlsx"

df = pd.read_excel(excel_file)

# 将数据保存为 CSV 文件
csv_file = 'data.csv'
df.to_csv(csv_file, index=False)

print(f"Excel 文件已成功转换为 {csv_file}")