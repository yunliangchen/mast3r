from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import numpy as np

# read the first frame of a mp4 file
import cv2
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
import h5py
import os
import random
from concurrent.futures import ProcessPoolExecutor
import tempfile
import shutil
import sys

# For DROID data format
def get_camera_extrinsic_matrix(calibration_6d):
    calibration_matrix = np.array(calibration_6d)
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    extrinsic_matrix = np.hstack((rotation_matrix, cam_pose.reshape(3, 1)))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
    return extrinsic_matrix

def get_ee_pose_matrix(cartesian_6d):
    cartesian_position = np.array(cartesian_6d)
    ee_pose = cartesian_position[:3]
    ee_euler = cartesian_position[3:]
    rotation_matrix = R.from_euler("XYZ", ee_euler).as_matrix()
    ee_pose_matrix = np.hstack((rotation_matrix, ee_pose.reshape(3, 1)))
    ee_pose_matrix = np.vstack((ee_pose_matrix, np.array([0, 0, 0, 1])))
    return ee_pose_matrix


def get_mid_poses(pose1, pose2):
    q1 = R.from_matrix(pose1[:3, :3].detach().cpu().numpy()).as_quat()
    q2 = R.from_matrix(pose2[:3, :3].detach().cpu().numpy()).as_quat()
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # Interpolate between the two quaternions
    times = np.array([0, 1])
    quaternions = np.array([q1, q2])
    # Create a Slerp interpolator
    slerp = Slerp(times, R.from_quat(quaternions))
    # Interpolate at the midpoint (0.5 for midpoint)
    mid_quat = slerp(0.5).as_quat()
    mid_rotation_matrix = R.from_quat(mid_quat).as_matrix()
    mid_translation = (pose1[:3, 3].detach().cpu().numpy() + pose2[:3, 3].detach().cpu().numpy()) / 2
    mid_pose = np.eye(4)
    mid_pose[:3, :3] = mid_rotation_matrix
    mid_pose[:3, 3] = mid_translation
    return mid_pose


# find relative poses
def get_relative_pose(pose1, pose2):
    pose1_inv = np.linalg.inv(pose1)
    relative_pose = np.dot(pose1_inv, pose2)
    return relative_pose

def angle_between_rotations(R1, R2):
        R_rel = np.dot(R1.T, R2)
        trace = np.trace(R_rel)
        angle = np.arccos((trace - 1) / 2)
        return angle * 180 / np.pi






def estimate_pose_mast3r(wrist_serial, ext1_serial, ext2_serial, temp_image_folder, no_stereo_flag=False):
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # you can put the path to a local checkpoint in model_name if needed
    # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    if not no_stereo_flag:
        images = load_images([os.path.join(temp_image_folder, f"tmp_left_frame_{wrist_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_left_frame_{ext1_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_left_frame_{ext2_serial}.jpg"),
                          os.path.join(temp_image_folder, f"tmp_right_frame_{wrist_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_right_frame_{ext1_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_right_frame_{ext2_serial}.jpg")], size=512)
    else:
        images = load_images([os.path.join(temp_image_folder, f"tmp_left_frame_{wrist_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_left_frame_{ext1_serial}.jpg"), os.path.join(temp_image_folder, f"tmp_left_frame_{ext2_serial}.jpg")], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # view1.keys(): ['img', 'true_shape', 'idx', 'instance']: torch.Size([N, 3, 288, 512]), tensor([[288, 512], ..., [288, 512]], dtype=torch.int32), [1, 2, 2, 0, 0, 1, ...] (len=N*(N-1)), ['1', '2', '2', '0', '0', '1', ...]
    # pred1.keys(): ['pts3d', 'conf', 'desc', 'desc_conf']: torch.Size([N, 288, 512, 3]), torch.Size([N, 288, 512]), torch.Size([N, 288, 512, 24]), torch.Size([N, 288, 512])
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals() # torch.Size([6, 1])
    poses = scene.get_im_poses() # torch.Size([6, 4, 4])
    pts3d = scene.get_pts3d() # 6 * torch.Size([288, 512, 3])
    confidence_masks = scene.get_masks() # 6 * torch.Size([288, 512])
    return scene, pts3d, imgs, confidence_masks

def compare_gt_mast3r(mast3r_poses, gt_poses, no_stereo_flag=False):
    """
    mast3r_poses: [wrist_left, ext1_left, ext2_left, wrist_right, ext1_right, ext2_right]
    gt_poses: [wrist_middle, ext1_middle, ext2_middle]
    """
    # All in 4x4 matrix format
    if no_stereo_flag:
        mid_pose_wrist = get_mid_poses(mast3r_poses[0], mast3r_poses[0])
        mid_pose_left = get_mid_poses(mast3r_poses[1], mast3r_poses[1])
        mid_pose_right = get_mid_poses(mast3r_poses[2], mast3r_poses[2])
    else:
        mid_pose_wrist = get_mid_poses(mast3r_poses[0], mast3r_poses[3])
        mid_pose_left = get_mid_poses(mast3r_poses[1], mast3r_poses[4])
        mid_pose_right = get_mid_poses(mast3r_poses[2], mast3r_poses[5])
    relative_pose_left_right = get_relative_pose(mid_pose_left, mid_pose_right)
    relative_pose_left_wrist = get_relative_pose(mid_pose_left, mid_pose_wrist)
    relative_pose_right_wrist = get_relative_pose(mid_pose_right, mid_pose_wrist)
    

    
    wrist_gt_mid_pose, ext1_gt_mid_pose, ext2_gt_mid_pose = gt_poses[0], gt_poses[1], gt_poses[2]
    relative_pose_left_right_gt = get_relative_pose(ext1_gt_mid_pose, ext2_gt_mid_pose)
    relative_pose_left_wrist_gt = get_relative_pose(ext1_gt_mid_pose, wrist_gt_mid_pose)
    relative_pose_right_wrist_gt = get_relative_pose(ext2_gt_mid_pose, wrist_gt_mid_pose)
    

    # compare the relative poses
    # print("relative_pose_left_right_gt:", relative_pose_left_right_gt)
    # print("relative_pose_left_right:", relative_pose_left_right)
    # print("relative_pose_left_wrist_gt:", relative_pose_left_wrist_gt)
    # print("relative_pose_left_wrist:", relative_pose_left_wrist)
    # print("relative_pose_right_wrist_gt:", relative_pose_right_wrist_gt)
    # print("relative_pose_right_wrist:", relative_pose_right_wrist)

    
    rotation_error_left_right = angle_between_rotations(relative_pose_left_right_gt[:3, :3], relative_pose_left_right[:3, :3])
    print("abs_angular_error Left Right:", rotation_error_left_right)
    rotation_error_left_wrist = angle_between_rotations(relative_pose_left_wrist_gt[:3, :3], relative_pose_left_wrist[:3, :3])
    print("abs_angular_error Left Wrist:", rotation_error_left_wrist)
    rotation_error_right_wrist = angle_between_rotations(relative_pose_right_wrist_gt[:3, :3], relative_pose_right_wrist[:3, :3])
    print("abs_angular_error Right Wrist:", rotation_error_right_wrist)

    translation_error_left_right = relative_pose_left_right[:3, 3] / np.linalg.norm(relative_pose_left_right[:3, 3]) - relative_pose_left_right_gt[:3, 3] / np.linalg.norm(relative_pose_left_right_gt[:3, 3])
    print("translation_error Left Right:", translation_error_left_right, np.linalg.norm(translation_error_left_right))
    translation_error_left_wrist = relative_pose_left_wrist[:3, 3] / np.linalg.norm(relative_pose_left_wrist[:3, 3]) - relative_pose_left_wrist_gt[:3, 3] / np.linalg.norm(relative_pose_left_wrist_gt[:3, 3])
    print("translation_error Left Wrist:", translation_error_left_wrist, np.linalg.norm(translation_error_left_wrist))
    translation_error_right_wrist = relative_pose_right_wrist[:3, 3] / np.linalg.norm(relative_pose_right_wrist[:3, 3]) - relative_pose_right_wrist_gt[:3, 3] / np.linalg.norm(relative_pose_right_wrist_gt[:3, 3])
    print("translation_error Right Wrist:", translation_error_right_wrist, np.linalg.norm(translation_error_right_wrist))

    errors = {"rotation_error_left_right": rotation_error_left_right, 
              "rotation_error_left_wrist": rotation_error_left_wrist, 
              "rotation_error_right_wrist": rotation_error_right_wrist, 
              "translation_error_left_right": translation_error_left_right.tolist(), 
              "translation_error_left_wrist": translation_error_left_wrist.tolist(), 
              "translation_error_right_wrist": translation_error_right_wrist.tolist(),
              "translation_error_left_right_norm": np.linalg.norm(translation_error_left_right),
              "translation_error_left_wrist_norm": np.linalg.norm(translation_error_left_wrist),
              "translation_error_right_wrist_norm": np.linalg.norm(translation_error_right_wrist)}

    return errors

    

def visualize(scene, pts3d, imgs, confidence_masks, visualize_matches=False):
    # visualize reconstruction
    scene.show()
    if visualize_matches:
        # find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # visualize a few matches
        import numpy as np
        from matplotlib import pyplot as pl
        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)


def process_traj_folder(traj_folder_path):
    # temp_image_folder = "/home/lawchen/project/droid/playground"
    # Create a unique temporary directory
    temp_image_folder = tempfile.mkdtemp()

    # find ".json" file in the folder
    # if file.endswith("errors.json") skip
    if len([file for file in os.listdir(traj_folder_path) if file.endswith("errors.json")]) > 0:
        print(f"Skipping {traj_folder_path}")
        # Clean up temporary files if desired
        shutil.rmtree(temp_image_folder)
        return
    try:
        json_path = [os.path.join(traj_folder_path, file) for file in os.listdir(traj_folder_path) if file.endswith(".json") and not file.endswith("errors.json")][0]
    except:
        print(f"Error with {traj_folder_path}")
        # Clean up temporary files if desired
        shutil.rmtree(temp_image_folder)
        return
    meta_data = json.load(open(json_path))
    trajectory_h5_path = os.path.join(traj_folder_path, "trajectory.h5")

    try:
        with h5py.File(trajectory_h5_path, 'r') as traj:
            
            wrist_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["wrist_cam_serial"]}_left'][()]
            wrist_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["wrist_cam_serial"]}_right'][()]
            ext1_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext1_cam_serial"]}_left'][()]
            ext1_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext1_cam_serial"]}_right'][()]
            ext2_cam_left_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext2_cam_serial"]}_left'][()]
            ext2_cam_right_extrinsics = traj['observation']['camera_extrinsics'][f'{meta_data["ext2_cam_serial"]}_right'][()]
    except:
        print(f"Error with {traj_folder_path}")
        # Clean up temporary files if desired
        shutil.rmtree(temp_image_folder)
        return


    wrist_cam_middle = (wrist_cam_left_extrinsics + wrist_cam_right_extrinsics) / 2
    ext1_cam_middle = (ext1_cam_left_extrinsics + ext1_cam_right_extrinsics) / 2
    ext2_cam_middle = (ext2_cam_left_extrinsics + ext2_cam_right_extrinsics) / 2

    # sample a timestep
    def compute_midpoints(X, k):
        midpoints = []
        for i in range(1, k):
            midpoint = ((2 * i - 1) * X) // (2 * k)
            midpoints.append(midpoint)
        return midpoints
    timesteps = compute_midpoints(len(wrist_cam_left_extrinsics), 5)
    timesteps = random.sample(range(0, len(wrist_cam_left_extrinsics)), 1)
    all_errors = {}
    for timestep in timesteps:
        print("Timestep", timestep)


        ext1_gt_mid_pose = get_camera_extrinsic_matrix(ext1_cam_middle[timestep])
        ext2_gt_mid_pose = get_camera_extrinsic_matrix(ext2_cam_middle[timestep])
        wrist_gt_mid_pose = get_camera_extrinsic_matrix(wrist_cam_middle[timestep])

        no_stereo_flag = False
        for cam_serial in [meta_data["wrist_cam_serial"], meta_data["ext1_cam_serial"], meta_data["ext2_cam_serial"]]:
            mp4_path = os.path.join(traj_folder_path, f"recordings/MP4/{cam_serial}-stereo.mp4")

            cap = cv2.VideoCapture(mp4_path)
            i = 0
            while True:
                ret, frame = cap.read()
                if i == timestep:
                    if not ret:
                        if not os.path.exists(mp4_path):
                            print(f"mp4_path {mp4_path} does not exist")
                            no_stereo_flag = True
                            break
                        else:
                            print(f"mp4_path {mp4_path} exists")
                            return
                if ret and i == timestep:
                    left_frame = frame[:, :1280]
                    right_frame = frame[:, 1280:]
                    cv2.imwrite(os.path.join(temp_image_folder, f"tmp_left_frame_{cam_serial}.jpg"), left_frame)
                    cv2.imwrite(os.path.join(temp_image_folder, f"tmp_right_frame_{cam_serial}.jpg"), right_frame)
                    
                    break
                i += 1
            cap.release()

            if no_stereo_flag:
                break

        if no_stereo_flag:
            for cam_serial in [meta_data["wrist_cam_serial"], meta_data["ext1_cam_serial"], meta_data["ext2_cam_serial"]]:
                mp4_path = os.path.join(traj_folder_path, f"recordings/MP4/{cam_serial}.mp4")

                cap = cv2.VideoCapture(mp4_path)
                i = 0
                while True:
                    ret, frame = cap.read()
                    if ret and i == timestep:
                        left_frame = frame[:, :1280]
                        cv2.imwrite(os.path.join(temp_image_folder, f"tmp_left_frame_{cam_serial}.jpg"), left_frame)
                        
                        break
                    i += 1
                    if i > len(wrist_cam_left_extrinsics):
                        return
                cap.release()


        scene, pts3d, imgs, confidence_masks = estimate_pose_mast3r(meta_data["wrist_cam_serial"], meta_data["ext1_cam_serial"], meta_data["ext2_cam_serial"], temp_image_folder, no_stereo_flag)
        if no_stereo_flag:
            wrist_gt_mid_pose, ext1_gt_mid_pose, ext2_gt_mid_pose = get_camera_extrinsic_matrix(wrist_cam_left_extrinsics[timestep]), get_camera_extrinsic_matrix(ext1_cam_left_extrinsics[timestep]), get_camera_extrinsic_matrix(ext2_cam_left_extrinsics[timestep])
        errors = compare_gt_mast3r(scene.get_im_poses(), [wrist_gt_mid_pose, ext1_gt_mid_pose, ext2_gt_mid_pose], no_stereo_flag)
        # visualize(scene, pts3d, imgs, confidence_masks, visualize_matches=True)
        all_errors[timestep] = errors

    # Clean up temporary files if desired
    shutil.rmtree(temp_image_folder)

    # save errors as json
    with open(os.path.join(traj_folder_path, "mast3r_camera_pose_estimation_errors.json"), 'w') as f:
        json.dump(all_errors, f)


def process_wrapper(traj_folder):

    # find all subfolders in the folder
    # traj_folder = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/AUTOLab/success/2023-07-23"
    traj_folder_paths = [os.path.join(traj_folder, folder) for folder in os.listdir(traj_folder) if os.path.isdir(os.path.join(traj_folder, folder))]
    # traj_folder_path = "/home/lawchen/project/droid/data/2023-07-23/Sun_Jul_23_20:53:26_2023"

    # Use ProcessPoolExecutor to process folders
    for traj_folder_path in traj_folder_paths:
        process_traj_folder(traj_folder_path)
    # num_workers = 2
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     executor.map(process_traj_folder, traj_folder_paths)
    print(f"Processed all trajectories in {traj_folder}")

def main(gpu_id, base_folders):
    # gpu_id = int(sys.argv[1])
    # folders = sys.argv[2:]
    
    # Set the GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # print(f"Processing folders {folders} on GPU {gpu_id}")
    # for folder in folders[1:]:
    #     process_wrapper(folder)
    for base_folder in base_folders:
        process_wrapper(base_folder)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", type=int)
    parser.add_argument("--base_folders", nargs='+', type=str)
    args = parser.parse_args()
    main(args.cuda_device, args.base_folders)