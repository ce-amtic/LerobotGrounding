import pandas as pd

path = "/pdata/oxe_lerobot/language_table/data/chunk-000/episode_001019.parquet"

df = pd.read_parquet(path)
print(df.columns)

# for i, row in df.iterrows():
#     print(row['observation.images.cam']['bytes'])
#     break