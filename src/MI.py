import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression

import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 读取CSV文件
# 假设CSV文件名为data.csv，路径根据实际情况修改
df = pd.read_csv('./data/transformed_data.csv', keep_default_na=False)


# 2. **方差分析（ANOVA）**：检验类型和房产价格之间是否有显著关系
# 将房产价格按照'Alley'类型分组
numeric_columns = df.select_dtypes(include='number').columns

results_mi = {}
for column in numeric_columns:
    mi = mutual_info_regression(df[[column]], df['SalePrice'])
    # 存储结果
    results_mi[column] = mi

sorted_map = dict(sorted(results_mi.items(), key=lambda item: item[1]))

# 创建保存文件的目录（如果不存在）
if not os.path.exists('out'):
    os.makedirs('out')

with open('./out/mi_result.txt', 'w') as f:
    # 遍历每个列及其对应的互信息
    for column, result in sorted_map.items():
        # 确保写入文件的内容都是合法字符，去除可能的空格等异常字符
        clean_column = column.strip()
        clean_result = str(result).strip()
        f.write(f"Column: {clean_column}\n")
        f.write(f"  mutual_info_regression: {clean_result}\n")
        f.write("-" * 50 + "\n")