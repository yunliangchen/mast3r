# read pose_errors_sorted_filtered_combined.csv and copy all folders listed in the "path" column to a new folder "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd"

import os
import pandas as pd
from tqdm import tqdm
df = pd.read_csv("pose_errors_sorted_filtered3.csv")
folders = df["path"].tolist()

for folder in tqdm(folders):
    os.system(f"cp -r {folder} /lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_5k")

