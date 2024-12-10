import pandas as pd
from io import StringIO
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

datapath = "data/train.csv"
data = pd.read_csv(datapath, keep_default_na=False)

#没有意义的缺失值进行清理
data = data.replace({'MasVnrType': 'NA', 'Electrical': 'NA', 'GarageYrBlt': 'NA', 'LotFrontage': 'NA'}, np.nan)

columns_to_check = ['MasVnrType', 'Electrical']
data = data.dropna(subset=columns_to_check)

# 将GarageYrBlt列的缺失值填充为-1
# 将LotFrontage列的缺失值填充为-1
data.fillna({'GarageYrBlt': -1, 'LotFrontage': -1}, inplace=True)

#数值列进行转换
data['LotFrontage'] = data['LotFrontage'].astype('Float64')
data['MasVnrArea'] = data['MasVnrArea'].astype('Float64')
data['GarageYrBlt'] = data['GarageYrBlt'].astype('Float64')






data = data.dropna(subset=['SalePrice'])

# Select target
y = data.SalePrice

predictors = data.drop(['SalePrice'], axis=1)

numeric_features = predictors.select_dtypes(include=['int64']).columns.tolist()
non_numeric_features = predictors.select_dtypes(include=['object']).columns.tolist()

numeric_stats = predictors[numeric_features].describe()
numeric_stats.to_csv("out/numeric_stats.csv")

# for feature in non_numeric_features:
#     num_unique_values = data[feature].nunique()
#     print(f"{feature}列的互异值数量为: {num_unique_values}")

#非数值列转换为数值标签
label_encoder = LabelEncoder()
label_mapping = {}
for feature in non_numeric_features:
    data[feature] = label_encoder.fit_transform(data[feature])
    label_mapping[feature] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    return obj

# 假设之前已经得到了label_mapping字典
label_mapping = convert_to_serializable(label_mapping)
data.to_csv('data/transformed_data.csv', index=False)
with open('data/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=4)

# print("非数值列已成功转换为数值标签，标签与值的对应关系已保存到label_mapping.json文件中。")


buffer = StringIO()

data.info(buf=buffer)

info_str = buffer.getvalue()
with open("out/new_data_info.txt", "w") as f:
    f.write(info_str)







