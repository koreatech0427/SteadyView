import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
from tqdm import tqdm

from config import (
    LOFTR_CONFIDENCE_THRESH,
    LOFTR_RESIZE_H,
    PHI_ABS_CLAMP,
    PHI_DELTA_CLAMP,
    PHI_MIN_INLIER_RATIO,
    PHI_MIN_MATCHES,
    SCENE_CUT_THRESHOLD,
)
from upright_model import infer_frame_angle

_LOFTR_CACHE = {}


def get_loftr_matcher(device):
    cache_key = str(device)
    matcher = _LOFTR_CACHE.get(cache_key)
    if matcher is None:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        matcher = LoFTR(pretrained='outdoor').to(device).eval()
        _LOFTR_CACHE[cache_key] = matcher
    return matcher


def detect_scene_cut(prev_gray, curr_gray, threshold=SCENE_CUT_THRESHOLD):
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray))) > threshold


def estimate_relative_phi_from_tracks(prev_gray, curr_gray, pa, pb, prev_phi=0.0):
    if detect_scene_cut(prev_gray, curr_gray):
        return 0.0
    if pa is None or pb is None or len(pa) < PHI_MIN_MATCHES:
        return prev_phi

    try:
        matrix, inliers = cv2.estimateAffinePartial2D(
            pa.reshape(-1, 1, 2),
            pb.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
    except cv2.error:
        return prev_phi

    if matrix is None or inliers is None:
        return prev_phi

    inlier_ratio = float(inliers.sum()) / max(len(inliers), 1)
    if inlier_ratio < PHI_MIN_INLIER_RATIO:
        return prev_phi

    phi = float(np.arctan2(matrix[0, 1], matrix[0, 0]) * 180.0 / np.pi)
    phi = float(np.clip(phi, -PHI_ABS_CLAMP, PHI_ABS_CLAMP))

    delta = phi - prev_phi
    if delta > PHI_DELTA_CLAMP:
        phi = prev_phi + PHI_DELTA_CLAMP
    elif delta < -PHI_DELTA_CLAMP:
        phi = prev_phi - PHI_DELTA_CLAMP
    return phi


def make_loftr_tensor(frame_bgr, new_w, new_h, device):
    gray = cv2.cvtColor(cv2.resize(frame_bgr, (new_w, new_h)), cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def analyze_video_shared(video_path, model, transform, device, mesh_size, demand, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_estimated = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError('Could not read the first frame.')

    h, w = first_frame.shape[:2]
    scale = LOFTR_RESIZE_H / h
    user_scale = scale
    new_h = (int(LOFTR_RESIZE_H) // 8) * 8
    new_w = (int(w * scale) // 8) * 8

    print('[Analyze] Initializing LoFTR...')
    matcher = get_loftr_matcher(device)

    prev_gray_small = cv2.cvtColor(cv2.resize(first_frame, (new_w, new_h)), cv2.COLOR_BGR2GRAY)
    prev_tensor = make_loftr_tensor(first_frame, new_w, new_h, device)

    theta0, sigma0 = infer_frame_angle(model, first_frame, transform, device)
    theta_os = [theta0]
    sigmas_sq = [sigma0]
    phis = []
    matched_pairs = []
    prev_phi = 0.0

    pbar_total = max(total_estimated - 1, 0) if total_estimated > 0 else None
    pbar = tqdm(total=pbar_total, desc='LoFTR + upright analysis', unit='frame')

    processed_pairs = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        theta_t, sigma_t = infer_frame_angle(model, frame, transform, device)
        theta_os.append(theta_t)
        sigmas_sq.append(sigma_t)

        curr_tensor = make_loftr_tensor(frame, new_w, new_h, device)
        with torch.inference_mode():
            correspondences = matcher({"image0": prev_tensor, "image1": curr_tensor})

        pa = correspondences['keypoints0'].cpu().numpy()
        pb = correspondences['keypoints1'].cpu().numpy()
        conf = correspondences['confidence'].cpu().numpy()
        good = conf > LOFTR_CONFIDENCE_THRESH
        pa = pa[good]
        pb = pb[good]

        if len(pa) > 0:
            pa[:, 0] *= (w * user_scale / new_w)
            pa[:, 1] *= (h * user_scale / new_h)
            pb[:, 0] *= (w * user_scale / new_w)
            pb[:, 1] *= (h * user_scale / new_h)

        matched_pairs.append((pa, pb))

        curr_gray_small = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2GRAY)
        phi_t = estimate_relative_phi_from_tracks(prev_gray_small, curr_gray_small, pa, pb, prev_phi)
        phis.append(phi_t)
        prev_phi = phi_t

        prev_tensor = curr_tensor
        prev_gray_small = curr_gray_small
        pbar.update(1)
        processed_pairs += 1
        if progress_callback is not None and pbar_total:
            percent = int(round(processed_pairs / pbar_total * 100))
            progress_callback(min(percent, 100), f"영상 분석 중... {processed_pairs}/{pbar_total}프레임")

    pbar.close()
    cap.release()

    n_frames = len(theta_os)
    scale_inv = 1.0 / user_scale
    rescaled_pairs = [
        (pa.astype(np.float32) * scale_inv, pb.astype(np.float32) * scale_inv)
        for pa, pb in matched_pairs
    ]

    if len(rescaled_pairs) != max(n_frames - 1, 0):
        raise RuntimeError('Frame analysis result lengths do not match.')

    info = {
        'width': w,
        'height': h,
        'scale': user_scale,
        'fps': fps,
        'n_frames': n_frames,
    }
    return theta_os, sigmas_sq, phis, rescaled_pairs, info
