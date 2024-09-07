from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import numpy as np
if __name__ == '__main__':
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
    # images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png', 'dust3r/croco/assets/Chateau1.png'], size=512)
    # images = load_images(['/home/lawchen/project/droid/playground/left_frame_24400334.jpg', '/home/lawchen/project/droid/playground/left_frame_22008760.jpg', '/home/lawchen/project/droid/playground/left_frame_18026681.jpg'], size=512)
    # images = load_images(['/home/lawchen/project/droid/playground/0723_left_frame_22008760.jpg', '/home/lawchen/project/droid/playground/0723_left_frame_24400334.jpg', '/home/lawchen/project/droid/playground/0723_left_frame_18026681.jpg'], size=512)
    images = load_images(['tmp_left_frame_18026681.jpg', 'tmp_left_frame_22008760.jpg', 'tmp_left_frame_24400334.jpg', 'tmp_right_frame_18026681.jpg', 
                          'tmp_right_frame_22008760.jpg', 'tmp_right_frame_24400334.jpg'], size=512)
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
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    # print("Loss:", loss)
    # print("Focals:", focals)
    print("Poses:", poses)


    # find the average of the stereo pair poses
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
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
    
    mid_pose_left = get_mid_poses(poses[1], poses[4])
    # print(mid_pose_left)

    mid_pose_right = get_mid_poses(poses[2], poses[5])

    # print(mid_pose_right)

    mid_pose_wrist = get_mid_poses(poses[0], poses[3])
    # print(mid_pose_wrist)


    # find relative poses
    def get_relative_pose(pose1, pose2):
        pose1_inv = np.linalg.inv(pose1)
        relative_pose = np.dot(pose1_inv, pose2)
        return relative_pose
    
    relative_pose_left_right = get_relative_pose(mid_pose_left, mid_pose_right)
    # print(relative_pose_left_right)

    relative_pose_left_wrist = get_relative_pose(mid_pose_left, mid_pose_wrist)
    # print(relative_pose_left_wrist)

    relative_pose_right_wrist = get_relative_pose(mid_pose_right, mid_pose_wrist)
    # print(relative_pose_right_wrist)

    """
    wrist_cam_extrinsics: 18026681
    [[ 0.44593979  0.84378732  0.29859784  0.45594187]
    [ 0.88844076 -0.45778868 -0.03320452 -0.09895442]
    [ 0.10867716  0.28009371 -0.95380123  0.14876992]
    [ 0.          0.          0.          1.        ]]
    ext1_cam_extrinsics: 22008760
    [[-0.92534965 -0.18470403  0.33107771  0.28599041]
    [-0.37860741  0.40505897 -0.83221611  0.58516839]
    [ 0.01960767 -0.89543936 -0.44475149  0.38252172]
    [ 0.          0.          0.          1.        ]]
    ext2_cam_extrinsics: 24400334
    [[ 0.74222416 -0.1709973   0.64796853  0.17787727]
    [-0.66519003 -0.30542575  0.68134964 -0.56333355]
    [ 0.08139733 -0.93673637 -0.34044007  0.30114697]
    [ 0.          0.          0.          1.        ]]
    """

    gt_pose_left = np.array([[-0.92534965, -0.18470403,  0.33107771,  0.28599041],
                             [-0.37860741,  0.40505897, -0.83221611,  0.58516839],
                            [ 0.01960767, -0.89543936, -0.44475149,  0.38252172],
                            [0, 0, 0, 1]])
    gt_pose_right = np.array([[ 0.74222416, -0.1709973,   0.64796853,  0.17787727],
                                [-0.66519003, -0.30542575,  0.68134964, -0.56333355],
                                [ 0.08139733, -0.93673637, -0.34044007,  0.30114697],
                                [0, 0, 0, 1]])
    gt_pose_wrist = np.array([[ 0.44593979,  0.84378732,  0.29859784,  0.45594187],
                                [ 0.88844076, -0.45778868, -0.03320452, -0.09895442],
                                [ 0.10867716,  0.28009371, -0.95380123,  0.14876992],
                                [0, 0, 0, 1]])

    relative_pose_left_right_gt = get_relative_pose(gt_pose_left, gt_pose_right)
    # print(relative_pose_left_right_gt)

    relative_pose_left_wrist_gt = get_relative_pose(gt_pose_left, gt_pose_wrist)
    # print(relative_pose_left_wrist_gt)

    relative_pose_right_wrist_gt = get_relative_pose(gt_pose_right, gt_pose_wrist)
    # print(relative_pose_right_wrist_gt)

    # compare the relative poses
    print("relative_pose_left_right_gt:", relative_pose_left_right_gt)
    print("relative_pose_left_right:", relative_pose_left_right)
    print("relative_pose_left_wrist_gt:", relative_pose_left_wrist_gt)
    print("relative_pose_left_wrist:", relative_pose_left_wrist)
    print("relative_pose_right_wrist_gt:", relative_pose_right_wrist_gt)
    print("relative_pose_right_wrist:", relative_pose_right_wrist)

    def angle_between_rotations(R1, R2):
        R_rel = np.dot(R1.T, R2)
        trace = np.trace(R_rel)
        angle = np.arccos((trace - 1) / 2)
        return angle * 180 / np.pi

    import roma
    import torch
    abs_angular_error = roma.rotmat_geodesic_distance(torch.tensor(relative_pose_left_right_gt[:3, :3]),
                                                      torch.tensor(relative_pose_left_right[:3, :3])) * 180 / np.pi
    print("abs_angular_error Left Right:", angle_between_rotations(relative_pose_left_right_gt[:3, :3], relative_pose_left_right[:3, :3]))
    print("abs_angular_error Left Right:", abs_angular_error)

    abs_angular_error = roma.rotmat_geodesic_distance(torch.tensor(relative_pose_left_wrist_gt[:3, :3]),
                                                        torch.tensor(relative_pose_left_wrist[:3, :3])) * 180 / np.pi
    print("abs_angular_error Left Wrist:", angle_between_rotations(relative_pose_left_wrist_gt[:3, :3], relative_pose_left_wrist[:3, :3]))
    print("abs_angular_error Left Wrist:", abs_angular_error)

    abs_angular_error = roma.rotmat_geodesic_distance(torch.tensor(relative_pose_right_wrist_gt[:3, :3]),
                                                        torch.tensor(relative_pose_right_wrist[:3, :3])) * 180 / np.pi
    print("abs_angular_error Right Wrist:", angle_between_rotations(relative_pose_right_wrist_gt[:3, :3], relative_pose_right_wrist[:3, :3]))
    print("abs_angular_error Right Wrist:", abs_angular_error)

    translation_error = relative_pose_left_right[:3, 3] / np.linalg.norm(relative_pose_left_right[:3, 3]) - relative_pose_left_right_gt[:3, 3] / np.linalg.norm(relative_pose_left_right_gt[:3, 3])
    print("translation_error Left Right:", translation_error, np.linalg.norm(translation_error))

    translation_error = relative_pose_left_wrist[:3, 3] / np.linalg.norm(relative_pose_left_wrist[:3, 3]) - relative_pose_left_wrist_gt[:3, 3] / np.linalg.norm(relative_pose_left_wrist_gt[:3, 3])
    print("translation_error Left Wrist:", translation_error, np.linalg.norm(translation_error))

    translation_error = relative_pose_right_wrist[:3, 3] / np.linalg.norm(relative_pose_right_wrist[:3, 3]) - relative_pose_right_wrist_gt[:3, 3] / np.linalg.norm(relative_pose_right_wrist_gt[:3, 3])
    print("translation_error Right Wrist:", translation_error, np.linalg.norm(translation_error))



    # visualize reconstruction
    scene.show()

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
