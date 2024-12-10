import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 读取CSV文件
# 假设CSV文件名为data.csv，路径根据实际情况修改
df = pd.read_csv('./data/transformed_data.csv', keep_default_na=False)


# 2. **方差分析（ANOVA）**：检验类型和房产价格之间是否有显著关系
# 将房产价格按照'Alley'类型分组
numeric_columns = df.select_dtypes(include='number').columns

# 对每个数值列进行方差分析
results_f = {}
results_h = {}

for column in numeric_columns:
    # 将数据按column分组，取出每个组的该列数据
    grouped = [df[df[column] == category]['SalePrice'] for category in df[column].unique()]
    
    # 进行单因素方差分析
    f_statistic, fp_value = stats.f_oneway(*grouped)
    # 使用Kruskal-Wallis检验比较不同类型的房产价格
    h_statistic, hp_value = stats.kruskal(*grouped)
    # 存储结果
    results_f[column] = {'F-statistic': f_statistic, 'P-value': fp_value}
    results_h[column] = {'H-statistic': h_statistic, 'P-value': hp_value}


# 绘制F-statistic结果的柱状图并保存
f_statistics = [results_f[col]['F-statistic'] for col in numeric_columns]
plt.figure(figsize=(15, 6))
plt.xticks(fontsize = 8)
sns.barplot(x=numeric_columns, y=f_statistics)
plt.title("F-statistic for Each Numeric Column")
plt.xlabel("Numeric Column")
plt.ylabel("F-statistic")
plt.xticks(rotation=45)
# 创建保存图片的目录（如果不存在）
if not os.path.exists('out/results_images'):
    os.makedirs('out/results_images')
# 保存图片
plt.savefig('out/results_images/f_statistic_result.png')
plt.close()

# 绘制H-statistic结果的柱状图并保存
h_statistics = [results_h[col]['H-statistic'] for col in numeric_columns]
plt.figure(figsize=(15, 6))
plt.xticks(fontsize = 8)
sns.barplot(x=numeric_columns, y=h_statistics)
plt.title("H-statistic for Each Numeric Column")
plt.xlabel("Numeric Column")
plt.ylabel("H-statistic")
plt.xticks(rotation=45)
# 保存图片
plt.savefig('out/results_images/h_statistic_result.png')
plt.close()

# 绘制F-statistic的pvalue结果的柱状图并保存
fP_value = [results_f[col]['P-value'] for col in numeric_columns]
plt.figure(figsize=(15, 6))
plt.xticks(fontsize = 8)
sns.barplot(x=numeric_columns, y=fP_value)
plt.title("F-statistic P-value for Each Numeric Column")
plt.xlabel("Numeric Column")
plt.ylabel("P-value")
plt.xticks(rotation=45)

# 保存图片
plt.savefig('out/results_images/f_statistic_p_result.png')
plt.close()

# 绘制H-statistic pvalue结果的柱状图并保存
hP_value = [results_h[col]['P-value'] for col in numeric_columns]
plt.figure(figsize=(15, 6))
plt.xticks(fontsize = 8)
sns.barplot(x=numeric_columns, y=hP_value)
plt.title("H-statistic P-value for Each Numeric Column")
plt.xlabel("Numeric Column")
plt.ylabel("P-value")
plt.xticks(rotation=45)
# 保存图片
plt.savefig('out/results_images/h_statistic_p_result.png')
plt.close()


# 按照results_f中p值从小到大排序并保存到文件
sorted_results_f = sorted(results_f.items(), key=lambda x: x[1]['P-value'])
# 创建保存文件的目录（如果不存在）
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')
with open('out/results_f_sorted.txt', 'w') as f:
    for item in sorted_results_f:
        feature_name = item[0]
        p_value = item[1]['P-value']
        f.write(f"{feature_name} {p_value}\n")

# 按照results_h中p值从小到大排序并保存到文件
sorted_results_h = sorted(results_h.items(), key=lambda x: x[1]['P-value'])
with open('out/results_h_sorted.txt', 'w') as f:
    for item in sorted_results_h:
        feature_name = item[0]
        p_value = item[1]['P-value']
        f.write(f"{feature_name} {p_value}\n")

print("结果已成功排序并保存到对应的txt文件中。")

'''
for column, result in results_f.items():
    print(f"Column: {column}")
    print(f"  F-statistic: {result['F-statistic']}")
    print(f"  P-value: {result['P-value']}")
    if result['P-value'] < 0.05:
        print(f"  => 该列与'SalePrice'类型之间存在显著差异")
    else:
        print(f"  => 该列与'SalePrice'类型之间没有显著差异")
    print("-" * 50)

for column, result in results_h.items():
    print(f"Column: {column}")
    print(f"  H-statistic: {result['H-statistic']}")
    print(f"  P-value: {result['P-value']}")
    if result['P-value'] < 0.05:
            print(f"  => 该列与'SalePrice'类型之间存在显著差异")
    else:
        print(f"  => 该列与'SalePrice'类型之间没有显著差异")
    print("-" * 50)
'''