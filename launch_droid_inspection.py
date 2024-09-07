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

def divide_folders_among_gpus(base_folder, num_gpus):
    all_folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    folders_per_gpu = len(all_folders) // num_gpus
    
    processes = []
    for gpu_id in range(num_gpus):
        time.sleep(3)
        start_idx = gpu_id * folders_per_gpu
        end_idx = (gpu_id + 1) * folders_per_gpu if gpu_id < num_gpus - 1 else len(all_folders)
        folders_for_gpu = all_folders[start_idx:end_idx]
        
        # Launch a subprocess for each GPU
        print(f'Launching subprocess for GPU {gpu_id} with {len(folders_for_gpu)} folders: {folders_for_gpu}')
        process = subprocess.Popen(['python', 'test_eval_error.py', str(gpu_id)] + folders_for_gpu)
        processes.append(process)
    
    for process in processes:
        process.wait()


def generate_shell_script(base_folder, num_gpus):
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
    folders_per_gpu = len(all_folders) // num_gpus
    print(all_folders)

    script = f"#!/bin/bash\n"
    for gpu_id in range(num_gpus):
        script += f"export CUDA_VISIBLE_DEVICES={gpu_id}\n"
        start_idx = gpu_id * folders_per_gpu
        end_idx = (gpu_id + 1) * folders_per_gpu if gpu_id < num_gpus - 1 else len(all_folders)
        folders_for_gpu = all_folders[start_idx:end_idx]
        script += f"python test_eval_error.py --base_folders "
        for folder in folders_for_gpu:
            script += f"{folder} "
        script += f"&\n"
    with open(f"run_gpu_1.sh", "w") as f:
        f.write(script)

if __name__ == '__main__':
    # /lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/{Lab}/success/{date}
    # Lab = {CLVR, ILIAD, IPRL, RAD, REAL, WEIRD, AUTOLab, GuptaLab, IRIS, PennPAL, RAIL, RPL, TRI}
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str)
    base_folder = parser.parse_args().base_folder
    # base_folder = '/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/ILIAD/success'
    num_gpus = 8
    # divide_folders_among_gpus(base_folder, num_gpus)
    generate_shell_script(base_folder, num_gpus)
