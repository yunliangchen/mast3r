
breakpoint()
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images


import numpy as np

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
import cv2

DEVICE = 'cuda'
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
BORDER = 3


def get_mast3r_output(img0, img1):
    # Load model, run inference
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    images = load_images([img0, img1], size=512)
    output = inference([tuple(images)], model, DEVICE,
                       batch_size=1, verbose=False)

    # raw predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=DEVICE, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= BORDER) & \
                        (matches_im0[:, 0] < int(W0) - BORDER) & \
                        (matches_im0[:, 1] >= BORDER) & \
                        (matches_im0[:, 1] < int(H0) - BORDER)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= BORDER) & \
                        (matches_im1[:, 0] < int(W1) - BORDER) & \
                        (matches_im1[:, 1] >= BORDER) & \
                        (matches_im1[:, 1] < int(H1) - BORDER)

    valid_matches = valid_matches_im0 & valid_matches_im1

    # matches are Nx2 image coordinates.
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    # Convert the other outputs to numpy arrays
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()

    return matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1

def scale_intrinsics(K: np.NDArray, prev_w: float, prev_h: float) -> np.NDArray:
    """Scale the intrinsics matrix by a given factor .

    Args:
        K (NDArray): 3x3 intrinsics matrix
        scale (float): Scale factor

    Returns:
        NDArray: Scaled intrinsics matrix
    """
    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = 512.0 / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = 384.0 / prev_h  # sizes of the images in the Mast3r dataset

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h

    return K_scaled

matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1 = get_mast3r_output('dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png') # call mast3r example script with custom images, all variables are np arrays

# use PnP from opencv
retval, rvec, tvec = cv2.solvePnP(
        objectPoints=pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :],
        imagePoints=matches_im1.astype(np.float32),    # ensure same datatype for opencv
        cameraMatrix=scale_K,                          # scale original intrinsics to Mast3r output resolution (512, 384)
        distCoeffs=np.zeros((0,)),                     # No distortion assumed
        flags=cv2.SOLVEPNP_EPNP)

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    breakpoint()
    # at this stage, you have the raw dust3r predictions
    # view1.keys(): ['img', 'true_shape', 'idx', 'instance']: torch.Size([1, 3, 384, 512]), tensor([[384, 512]], dtype=torch.int32), [0], ['0']
    # pred1.keys(): ['pts3d', 'conf', 'desc', 'desc_conf']: torch.Size([1, 384, 512, 3]), torch.Size([1, 384, 512]), torch.Size([1, 384, 512, 24]), torch.Size([1, 384, 512])
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)