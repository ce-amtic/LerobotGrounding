import pandas as pd

# /pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds/data/chunk-000/episode_000001.parquet

path = "/pdata/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds/data/chunk-000/episode_000001.parquet"

df = pd.read_parquet(path)
print(df.columns)