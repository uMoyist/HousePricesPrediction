datapath = "data/train.csv"
data = pd.read_csv(datapath, keep_default_na=False)

#没有意义的缺失值进行清理
data = data.replace({'MasVnrType': 'NA', 'Electrical': 'NA'}, np.nan)

columns_to_check = ['MasVnrType', 'Electrical']
data = data.dropna(subset=columns_to_check)

# 将GarageYrBlt列的缺失值填充为-1
# 将LotFrontage列的缺失值填充为-1
data.fillna({'GarageYrBlt': -1, 'LotFrontage': -1}, inplace=True)

#数值列进行转换
data['LotFrontage'] = data['LotFrontage'].astype('Float64')
data['MasVnrArea'] = data['MasVnrArea'].astype('Float64')
data['GarageYrBlt'] = data['GarageYrBlt'].astype('Float64')