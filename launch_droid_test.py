import subprocess
import multiprocessing
import os
import subprocess
import argparse
import time

def compute_trajs_remaining(subfolder):
    i = 0
    j = 0
    traj_folder_paths = [os.path.join(subfolder, folder) for folder in os.listdir(os.path.join(subfolder)) if os.path.isdir(os.path.join(subfolder, folder))]
    for traj_folder_path in traj_folder_paths:
        try:
            if len([file for file in os.listdir(traj_folder_path) if file.endswith("errors.json")]) > 0:
                i += 1
            elif len([os.path.join(traj_folder_path, file) for file in os.listdir(traj_folder_path) if file.endswith(".json") and not file.endswith("errors.json")]) > 0:
                j += 1
        except:
            print(f"Error in {traj_folder_path}")
    return i, j

def find_folders(base_folder, num_gpus=8):
    all_folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    print(f"Total folders: {len(all_folders)}")
    folders_to_remove = []
    for folder in all_folders:

        jobs_done, jobs_remaining = compute_trajs_remaining(folder)
        if jobs_remaining < 10:
            folders_to_remove.append(folder)
        else:
            print(folder, jobs_remaining)
    for folder in folders_to_remove:
        all_folders.remove(folder)
    print(f"Total folders remaining: {len(all_folders)}")
    
    # add to a list of 8
    if len(all_folders) <= num_gpus:
        folders = all_folders
    else:
        folders = all_folders[:num_gpus]
        # add one by one the remaining folders
        idx = 0
        for folder in all_folders[num_gpus:]:
            folders[idx % num_gpus] += f' {folder}'
            idx += 1
    print(folders)
    return folders
def run_process(gpu_id, base_folder):
    base_folders_list = base_folder.split()
    subprocess.run([
        'python', 'test_eval_error.py',
        '--base_folders', *base_folders_list,
        '--cuda_device', str(gpu_id)
    ])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str)
    args = parser.parse_args()
    folders = find_folders(args.base_folder)
    processes = []
    for gpu_id, folder in enumerate(folders):
        p = multiprocessing.Process(target=run_process, args=(gpu_id, folder))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
