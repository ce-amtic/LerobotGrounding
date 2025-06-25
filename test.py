import pandas as pd

path = "/pdata/sim_tabletop_tasks_lerobot_0617/data/chunk-000/episode_000168.parquet"

df = pd.read_parquet(path) #, columns=["sub_task_index"])
print(df.columns)

# for i, row in df.iterrows():
#     print(row['observation.images.cam']['bytes'])
#     break