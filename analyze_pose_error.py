# go through 
# /lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/{Lab}/success/{date}
# Lab = {CLVR, ILIAD, IPRL, RAD, REAL, WEIRD, AUTOLab, GuptaLab, IRIS, PennPAL, RAIL, RPL, TRI}
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
path_error_dict = {"path":[],
                "rotation_error_left_right": [], 
                "rotation_error_left_wrist": [], 
                "rotation_error_right_wrist": [], 
                "translation_error_left_right_norm": [],
                "translation_error_left_wrist_norm": [],
                "translation_error_right_wrist_norm": []}


ALL_LABS = ["CLVR", "ILIAD", "IPRL", "RAD", "REAL", "WEIRD", "AUTOLab", "GuptaLab", "IRIS", "PennPAL", "RAIL", "RPL", "TRI"]
for lab in ALL_LABS:
    base_folder = f'/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/{lab}/success'
    all_folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    all_errors = {"rotation_error_left_right": [], 
                "rotation_error_left_wrist": [], 
                "rotation_error_right_wrist": [], 
                "translation_error_left_right_norm": [],
                "translation_error_left_wrist_norm": [],
                "translation_error_right_wrist_norm": []}

    
    # errors = {"rotation_error_left_right": rotation_error_left_right, 
    #             "rotation_error_left_wrist": rotation_error_left_wrist, 
    #             "rotation_error_right_wrist": rotation_error_right_wrist, 
    #             "translation_error_left_right": translation_error_left_right.tolist(), 
    #             "translation_error_left_wrist": translation_error_left_wrist.tolist(), 
    #             "translation_error_right_wrist": translation_error_right_wrist.tolist(),
    #             "translation_error_left_right_norm": np.linalg.norm(translation_error_left_right),
    #             "translation_error_left_wrist_norm": np.linalg.norm(translation_error_left_wrist),
    #             "translation_error_right_wrist_norm": np.linalg.norm(translation_error_right_wrist)}
    

    i_ = 0
    j_ = 0
    for subfolder in all_folders:
        k_ = 0
        traj_folder_paths = [os.path.join(base_folder, subfolder, folder) for folder in os.listdir(os.path.join(base_folder, subfolder)) if os.path.isdir(os.path.join(base_folder, subfolder, folder))]
        for traj_folder_path in traj_folder_paths:
            try:
                if len([file for file in os.listdir(traj_folder_path) if file.endswith("errors.json")]) > 0:
                    # load json
                    errors = json.load(open(os.path.join(traj_folder_path, 'mast3r_camera_pose_estimation_errors.json')))
                    i_ += 1
                    path_error_dict["path"].append(traj_folder_path)
                    for key in all_errors:
                        timestep = list(errors.keys())[0]
                        all_errors[key].append(errors[timestep][key])
                        path_error_dict[key].append(errors[timestep][key])
                elif len([os.path.join(traj_folder_path, file) for file in os.listdir(traj_folder_path) if file.endswith(".json") and not file.endswith("errors.json")]) > 0:
                    j_ += 1
                    k_ += 1
            except:
                print(f"Error in {traj_folder_path}")
        # if k_ > 10:
        #     print(f"{subfolder} has {k_} trajectories remaining")
    print(f"Lab {lab}, Processed {i_} trajectories")
    # print(f"Lab {lab}, {j_} trajectories remaining")

    # # plot histograms
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # axs = axs.flatten()
    # for i, key in enumerate(all_errors):
    #     axs[i].hist(all_errors[key], bins=20)
    #     axs[i].set_title(key)

    # # add title
    # fig.suptitle(f'Pose error histograms {lab}', fontsize=16)
    # plt.savefig(f'pose_error_histograms_{lab}.png')


    # count the number of trajectories whose:
    # - translation error is less than 0.1
    # - rotation error is less than 10 degrees
    # rotation_count = 0
    # translation_count = 0
    # both_count = 0
    # for i in range(len(all_errors["rotation_error_left_right"])):
    #     if (all_errors["rotation_error_left_right"][i] < 10 or all_errors["rotation_error_left_right"][i] > 170) \
    #         and (all_errors["rotation_error_left_wrist"][i] < 10 or all_errors["rotation_error_left_wrist"][i] > 170) \
    #             and (all_errors["rotation_error_right_wrist"][i] < 10 or all_errors["rotation_error_right_wrist"][i] > 170):
    #         rotation_count += 1
    #     if np.mean([all_errors["translation_error_left_right_norm"][i], all_errors["translation_error_left_wrist_norm"][i], all_errors["translation_error_right_wrist_norm"][i]]) < 0.1:
    #         translation_count += 1
    #     if (all_errors["rotation_error_left_right"][i] < 10 or all_errors["rotation_error_left_right"][i] > 170) \
    #         and (all_errors["rotation_error_left_wrist"][i] < 10 or all_errors["rotation_error_left_wrist"][i] > 170) \
    #             and (all_errors["rotation_error_right_wrist"][i] < 10 or all_errors["rotation_error_right_wrist"][i] > 170) \
    #                 and np.mean([all_errors["translation_error_left_right_norm"][i], all_errors["translation_error_left_wrist_norm"][i], all_errors["translation_error_right_wrist_norm"][i]]) < 0.1:
    #         both_count += 1
    # if i_ == 0:
    #     continue
    # print(f"Lab {lab}, Rotation error < 10 or > 170: {rotation_count/i_}")
    # print(f"Lab {lab}, Translation error < 0.1: {translation_count/i_}")
    # print(f"Lab {lab}, Both: {both_count/i_}")



# convert path_error_dict to a pandas dataframe
df = pd.DataFrame(path_error_dict)
df.to_csv('pose_errors.csv')
print(df.head())
"""


# # read the csv
# df = pd.read_csv('pose_errors.csv')
# df['rotation_error_left_right'] = df['rotation_error_left_right'].apply(lambda x: min(x, 180-x))
# df['rotation_error_left_wrist'] = df['rotation_error_left_wrist'].apply(lambda x: min(x, 180-x))
# df['rotation_error_right_wrist'] = df['rotation_error_right_wrist'].apply(lambda x: min(x, 180-x))
# # compute max rotation and translation error
# df['max_rotation_error'] = df[['rotation_error_left_right', 'rotation_error_left_wrist', 'rotation_error_right_wrist']].max(axis=1)
# df['max_translation_error'] = df[['translation_error_left_right_norm', 'translation_error_left_wrist_norm', 'translation_error_right_wrist_norm']].max(axis=1)
# # sort by max_rotation_error and then max_translation_error from lowest to highest
# df = df.sort_values(by=['max_rotation_error', 'max_translation_error'])
# # save as sorted
# df.to_csv('pose_errors_sorted.csv', index=False)

# # find out how many satisfies: rotation_error_left_right < 10 and max_rotation_error < 15
# df = pd.read_csv('pose_errors_sorted.csv')
# df = df[df['rotation_error_left_right'] < 10]
# df = df[df['max_rotation_error'] < 15]
# df.to_csv('pose_errors_sorted_filtered.csv', index=False)

# df = pd.read_csv('pose_errors_sorted_filtered.csv')
# print the row whose rotation_error_left_right is max
# print(df.iloc[df['rotation_error_left_right'].idxmax()])

# # find out how many satisfies: rotation_error_left_right < 5
df = pd.read_csv('pose_errors_sorted.csv')
df = df[df['rotation_error_left_right'] < 3.5]
df.to_csv('pose_errors_sorted_filtered2.csv', index=False)

# combine pose_errors_sorted_filtered and pose_errors_sorted_filtered2
# i.e., find out how many satisfies: rotation_error_left_right < 5 and max_rotation_error < 15 or rotation_error_left_right < 10 and max_rotation_error < 15
# df = pd.read_csv('pose_errors_sorted.csv')
# df = df[(df['rotation_error_left_right'] < 5) | (df['rotation_error_left_right'] < 10) & (df['max_rotation_error'] < 15)]
# df.to_csv('pose_errors_sorted_filtered_combined.csv', index=False)