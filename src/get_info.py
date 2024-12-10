import pandas as pd
from io import StringIO


datapath = "../data/train.csv"
dataset = pd.read_csv(datapath)

buffer = StringIO()

dataset.info(buf=buffer)

info_str = buffer.getvalue()
with open("../out/data_info.txt", "w") as f:
    f.write(info_str)

dataset.describe().to_csv("../out/data_describe.csv")
