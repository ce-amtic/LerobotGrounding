import pandas as pd

path = "/pdata/oxe_lerobot/bc_z/data/chunk-000/episode_044228.parquet"

df = pd.read_parquet(path)
print(df.columns)

for i, row in df.iterrows():
    print(row['observation.images.cam']['bytes'])
    break