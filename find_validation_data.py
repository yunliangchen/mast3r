# find folders in pose_errors_sorted_filtered.csv but not in pose_errors_sorted_filtered2.csv

import os
import pandas as pd
from tqdm import tqdm
import glob
import json
# df1 = pd.read_csv("/lustre/fsw/portfolios/nvr/users/lawchen/project/mast3r/pose_errors_sorted_filtered.csv")
# folders = df1["path"].tolist()

# df2 = pd.read_csv("/lustre/fsw/portfolios/nvr/users/lawchen/project/mast3r/pose_errors_sorted_filtered2.csv")
# folders2 = df2["path"].tolist()

# # filter df1 to only include folders not in df2
# df1 = df1[~df1["path"].isin(folders2)]
# df1.to_csv("/lustre/fsw/portfolios/nvr/users/lawchen/project/mast3r/pose_errors_sorted_filtered3.csv", index=False)

df3 = pd.read_csv("/lustre/fsw/portfolios/nvr/users/lawchen/project/mast3r/pose_errors_sorted_filtered3.csv")
language_file_path="/lustre/fsw/portfolios/nvr/users/lawchen/project/droid/droid/aggregated-annotations-030724.json"
with open(language_file_path, 'r') as file:
    annotations = json.load(file)
path_with_annotations = []
# for each of the path, check if there is an annotation file
for path in tqdm(df3["path"].tolist()):
    # check if there is a metadata file
    metadata_files = glob.glob(os.path.join(path, "metadata*.json"))
    if len(metadata_files) == 0:
        print("No metadata file found in", path)
        continue
    metadata_file = metadata_files[0]
    # extract "TRI+52ca9b6a+2023-11-07-15h-30m-09s" from "metadata_TRI+52ca9b6a+2023-11-07-15h-30m-09s.json"
    metadata_name = os.path.basename(metadata_file).split(".")[0]
    metadata_name = metadata_name.split("_")[1]
    if metadata_name in annotations.keys():
        path_with_annotations.append(path)

df3 = df3[df3["path"].isin(path_with_annotations)]
df3.to_csv("/lustre/fsw/portfolios/nvr/users/lawchen/project/mast3r/pose_errors_sorted_filtered3.csv", index=False)