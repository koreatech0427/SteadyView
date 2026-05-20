import cv2
import math
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from tqdm import tqdm

from config import (
    LAMBDA,
    PATCH_CROP_RATIO,
    PHI_ABS_CLAMP,
    PHI_DELTA_CLAMP,
    PHI_MIN_INLIER_RATIO,
    PHI_MIN_MATCHES,
    SCENE_CUT_THRESHOLD,
    TAU,
)
from upright_model import build_transform, infer_frame_angle, load_upright_model


PHI_MAX_WIDTH = 640
FLOW_MAX_CORNERS = 600
FLOW_QUALITY_LEVEL = 0.01
FLOW_MIN_DISTANCE = 8
FLOW_BLOCK_SIZE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3

UPRIGHT_ONLY_MIN_SIGMA_SQ = 16.0
UPRIGHT_ONLY_MAX_SIGMA_SQ = 64.0

ENABLE_RAW_THETA_OUTLIER_FILTER = True
ENABLE_FINAL_ANGLE_SMOOTHING = True
THETA_OUTLIER_WINDOW = 11
THETA_OUTLIER_THRESHOLD = 2.0
FINAL_SMOOTH_WINDOW = 9
FINAL_SMOOTH_ALPHA = 0.20

ENABLE_STABILIZED_INPUT_MODE = True
STABILIZED_THETA_CLAMP = None
STABILIZED_MAX_FRAME_DELTA = None
STABILIZED_FINAL_SMOOTH_WINDOW = 15
STABILIZED_FINAL_SMOOTH_ALPHA = 0.12

CROP_SAMPLE_STEP = 1
CROP_ERODE_ITER = 5
CROP_EVAL_SCALE = 0.5

# Upright-only 크롭 비율 조정 위치:
# CROP_BORDER_MARGIN은 기본 안전 여백입니다.
# CROP_ARTIFACT_MARGIN을 키우면 검은 테두리/외곽 왜곡은 줄지만 화면이 더 확대됩니다.
# 렌더 단계의 guard_x/guard_y 비율(아래 render_upright_video의 0.035)도 추가 확대에 영향을 줍니다.
CROP_BORDER_MARGIN = 0.02
CROP_ARTIFACT_MARGIN = 0.10


def run_upright_adjustment(
    video_path,
    model_path,
    output_path,
    tau=TAU,
    lam=LAMBDA,
    auto_crop=True,
    crop_sample_step=CROP_SAMPLE_STEP,
    crop_erode_iter=CROP_ERODE_ITER,
    crop_eval_scale=CROP_EVAL_SCALE,
    crop_border_margin=CROP_BORDER_MARGIN,
    progress_callback=None,
):
    def report(progress, message):
        if progress_callback is not None:
            progress_callback(int(progress), message)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Start] upright-only device: {device}')
    report(1, 'Preparing upright model.')

    model = load_upright_model(model_path, device)
    transform = build_transform()
    report(5, 'Upright model loaded.')

    theta_os, sigmas_sq, phis, info = analyze_angles(
        video_path,
        model,
        transform,
        device,
        progress_callback=lambda percent, message: report(5 + round(percent * 0.55), message),
    )

    n_frames = len(theta_os)
    if len(phis) < max(n_frames - 1, 0):
        phis = phis + [0.0] * (n_frames - 1 - len(phis))
    phis = phis[:n_frames - 1]

    if ENABLE_RAW_THETA_OUTLIER_FILTER:
        theta_os, sigmas_sq, outlier_count = replace_theta_outliers(theta_os, sigmas_sq)
        print(f'[Angles] corrected raw outliers: {outlier_count}')

    report(62, 'Optimizing upright angle path.')
    final_thetas = optimize_video_angles(theta_os, sigmas_sq, phis, lam=lam)
    output_thetas = truncate_angles(final_thetas, tau=tau)

    if ENABLE_FINAL_ANGLE_SMOOTHING:
        output_thetas = smooth_final_angles(output_thetas)
    if ENABLE_STABILIZED_INPUT_MODE:
        output_thetas = postprocess_stabilized_input_angles(output_thetas)

    report(70, 'Preparing upright render.')
    render_upright_video(
        video_path,
        output_path,
        output_thetas,
        info,
        auto_crop=auto_crop,
        crop_sample_step=crop_sample_step,
        crop_erode_iter=crop_erode_iter,
        crop_eval_scale=crop_eval_scale,
        crop_border_margin=crop_border_margin,
        crop_progress_callback=lambda percent, message: report(70 + round(percent * 0.10), message),
        render_progress_callback=lambda percent, message: report(80 + round(percent * 0.20), message),
    )

    corrections = np.abs(output_thetas)
    print('\n[Upright Stats]')
    print(f'  mean correction: {corrections.mean():.3f}')
    print(f'  max correction:  {corrections.max():.3f}')
    print(f'  >1 degree:       {(corrections > 1.0).sum()} frames ({(corrections > 1.0).mean() * 100:.1f}%)')
    print(f'  >5 degree:       {(corrections > 5.0).sum()} frames ({(corrections > 5.0).mean() * 100:.1f}%)')
    report(100, 'Upright correction completed.')


def analyze_angles(video_path, model, transform, device, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    theta_os = []
    sigmas_sq = []
    phis = []
    prev_gray_small = None
    prev_phi = 0.0
    processed = 0

    for frame_idx in tqdm(range(total), desc='upright optical-flow analysis', unit='frame'):
        ret, frame = cap.read()
        if not ret:
            break

        theta, sigma_sq = infer_frame_angle(model, frame, transform, device)
        theta_os.append(theta)
        sigmas_sq.append(float(np.clip(sigma_sq, UPRIGHT_ONLY_MIN_SIGMA_SQ, UPRIGHT_ONLY_MAX_SIGMA_SQ)))

        gray_small = resize_gray_for_phi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if prev_gray_small is None:
            phi = 0.0
        else:
            phi = estimate_relative_phi(prev_gray_small, gray_small, prev_phi)
            phis.append(phi)
            prev_phi = phi
        prev_gray_small = gray_small

        processed += 1
        if progress_callback is not None and total > 0:
            percent = int(round(processed / total * 100))
            progress_callback(min(percent, 100), f'Analyzing upright angles... {processed}/{total} frames')

    cap.release()
    if not theta_os:
        raise RuntimeError('Could not read any video frames.')

    info = {
        'width': width,
        'height': height,
        'fps': fps,
        'n_frames': len(theta_os),
    }
    return theta_os, sigmas_sq, phis, info


def resize_gray_for_phi(gray, max_width=PHI_MAX_WIDTH):
    h, w = gray.shape[:2]
    if w <= max_width:
        return gray
    scale = max_width / float(w)
    return cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def detect_scene_cut(prev_gray, curr_gray, threshold=SCENE_CUT_THRESHOLD):
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray))) > threshold


def estimate_relative_phi(prev_gray, curr_gray, prev_phi=0.0):
    if detect_scene_cut(prev_gray, curr_gray):
        return 0.0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=FLOW_MAX_CORNERS,
        qualityLevel=FLOW_QUALITY_LEVEL,
        minDistance=FLOW_MIN_DISTANCE,
        blockSize=FLOW_BLOCK_SIZE,
    )
    if prev_pts is None or len(prev_pts) < PHI_MIN_MATCHES:
        return prev_phi

    try:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=FLOW_WIN_SIZE,
            maxLevel=FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    except cv2.error:
        return prev_phi

    if curr_pts is None or status is None:
        return prev_phi

    status = status.reshape(-1).astype(bool)
    src_pts = prev_pts.reshape(-1, 2)[status]
    dst_pts = curr_pts.reshape(-1, 2)[status]
    if len(src_pts) < PHI_MIN_MATCHES:
        return prev_phi

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None or inliers is None:
        return prev_phi

    inlier_ratio = float(inliers.sum()) / max(len(inliers), 1)
    if inlier_ratio < PHI_MIN_INLIER_RATIO:
        return prev_phi

    phi = float(np.arctan2(matrix[0, 1], matrix[0, 0]) * 180.0 / np.pi)
    phi = float(np.clip(phi, -PHI_ABS_CLAMP, PHI_ABS_CLAMP))
    delta = phi - prev_phi
    if delta > PHI_DELTA_CLAMP:
        return prev_phi + PHI_DELTA_CLAMP
    if delta < -PHI_DELTA_CLAMP:
        return prev_phi - PHI_DELTA_CLAMP
    return phi


def optimize_video_angles(theta_os, sigmas_sq, phis, lam=LAMBDA):
    theta_os = np.array(theta_os, dtype=np.float64)
    sigmas_sq = np.array(sigmas_sq, dtype=np.float64)
    phis = np.array(phis, dtype=np.float64)

    def residuals(theta):
        term1 = (1.0 / np.sqrt(sigmas_sq + 1e-8)) * (theta - theta_os)
        term2 = np.sqrt(lam) * ((theta[1:] - theta[:-1]) - phis)
        return np.concatenate([term1, term2])

    return least_squares(residuals, x0=theta_os.copy(), method='lm').x


def truncate_angles(thetas, tau=TAU):
    return np.sign(thetas) * np.minimum(np.abs(thetas), tau)


def median_filter_1d(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or window <= 1:
        return values.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    filtered = np.empty_like(values)
    for i in range(len(values)):
        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)
        filtered[i] = np.median(values[left:right])
    return filtered


def replace_theta_outliers(theta_os, sigmas_sq, window=THETA_OUTLIER_WINDOW, threshold=THETA_OUTLIER_THRESHOLD):
    theta = np.asarray(theta_os, dtype=np.float64)
    sigmas = np.asarray(sigmas_sq, dtype=np.float64)
    baseline = median_filter_1d(theta, window)
    outlier_mask = np.abs(theta - baseline) > threshold
    if np.any(outlier_mask):
        theta = theta.copy()
        sigmas = sigmas.copy()
        theta[outlier_mask] = baseline[outlier_mask]
        sigmas[outlier_mask] = np.maximum(sigmas[outlier_mask], 4.0)
    return theta.tolist(), sigmas.tolist(), int(outlier_mask.sum())


def smooth_final_angles(angles, window=FINAL_SMOOTH_WINDOW, alpha=FINAL_SMOOTH_ALPHA):
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) == 0:
        return angles

    median_smoothed = median_filter_1d(angles, window)
    smoothed = np.empty_like(median_smoothed)
    smoothed[0] = median_smoothed[0]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, len(median_smoothed)):
        smoothed[i] = alpha * median_smoothed[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def limit_angle_step(angles, max_delta=STABILIZED_MAX_FRAME_DELTA):
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) == 0 or max_delta is None:
        return angles.copy()

    limited = np.empty_like(angles)
    limited[0] = angles[0]
    max_delta = abs(float(max_delta))
    for i in range(1, len(angles)):
        delta = float(np.clip(angles[i] - limited[i - 1], -max_delta, max_delta))
        limited[i] = limited[i - 1] + delta
    return limited


def postprocess_stabilized_input_angles(angles):
    angles = np.asarray(angles, dtype=np.float64)
    if STABILIZED_THETA_CLAMP is not None:
        angles = np.clip(angles, -STABILIZED_THETA_CLAMP, STABILIZED_THETA_CLAMP)
    angles = smooth_final_angles(
        angles,
        window=STABILIZED_FINAL_SMOOTH_WINDOW,
        alpha=STABILIZED_FINAL_SMOOTH_ALPHA,
    )
    angles = limit_angle_step(angles, max_delta=STABILIZED_MAX_FRAME_DELTA)
    if STABILIZED_THETA_CLAMP is not None:
        angles = np.clip(angles, -STABILIZED_THETA_CLAMP, STABILIZED_THETA_CLAMP)
    return angles


def compute_rotation_source_map(angle_deg, src_w, src_h, out_width, out_height,
                                crop_x1=0, crop_y1=0, coord_width=None, coord_height=None):
    coord_w = coord_width if coord_width is not None else src_w
    coord_h = coord_height if coord_height is not None else src_h
    xs = np.linspace(crop_x1, crop_x1 + coord_w, out_width, dtype=np.float32)
    ys = np.linspace(crop_y1, crop_y1 + coord_h, out_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    matrix = cv2.getRotationMatrix2D((src_w / 2.0, src_h / 2.0), -float(angle_deg), 1.0)
    inv_matrix = cv2.invertAffineTransform(matrix)
    map_x = inv_matrix[0, 0] * grid_x + inv_matrix[0, 1] * grid_y + inv_matrix[0, 2]
    map_y = inv_matrix[1, 0] * grid_x + inv_matrix[1, 1] * grid_y + inv_matrix[1, 2]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def compute_auto_crop_box(angles, width, height, sample_step=CROP_SAMPLE_STEP,
                          erode_iter=CROP_ERODE_ITER, eval_scale=CROP_EVAL_SCALE,
                          border_margin=CROP_BORDER_MARGIN, progress_callback=None):
    eval_w = max(128, int(round(width * eval_scale)))
    eval_h = max(128, int(round(height * eval_scale)))
    common_valid = np.ones((eval_h, eval_w), dtype=bool)
    n_frames = len(angles)

    frame_indices = list(range(0, n_frames, max(1, sample_step)))
    if not frame_indices or frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    source_margin = 8.0
    src_x_min, src_x_max = source_margin, width - 1.0 - source_margin
    src_y_min, src_y_max = source_margin, height - 1.0 - source_margin

    total_samples = len(frame_indices)
    for sample_index, t in enumerate(tqdm(frame_indices, desc='upright auto-crop', unit='frame'), start=1):
        map_x, map_y = compute_rotation_source_map(
            angles[t],
            width,
            height,
            out_width=eval_w,
            out_height=eval_h,
            coord_width=width,
            coord_height=height,
        )
        valid = (
            np.isfinite(map_x) & np.isfinite(map_y) &
            (map_x >= src_x_min) & (map_x <= src_x_max) &
            (map_y >= src_y_min) & (map_y <= src_y_max)
        )
        if erode_iter > 0:
            valid = binary_erosion(valid, iterations=erode_iter)
        common_valid &= valid
        if progress_callback is not None:
            percent = int(round(sample_index / total_samples * 100))
            progress_callback(min(percent, 100), f'Calculating auto-crop... {sample_index}/{total_samples} samples')

    ys, xs = np.where(common_valid)
    if len(xs) == 0 or len(ys) == 0:
        print('[Crop] auto-crop failed; using full frame.')
        return 0, width, 0, height

    sx, sy = width / float(eval_w), height / float(eval_h)
    margin_x = int(np.ceil(width * border_margin))
    margin_y = int(np.ceil(height * border_margin))
    artifact_margin_x = max(margin_x, int(np.ceil(width * CROP_ARTIFACT_MARGIN)))
    artifact_margin_y = max(margin_y, int(np.ceil(height * CROP_ARTIFACT_MARGIN)))

    x1 = max(0, int(np.floor(xs.min() * sx)) + artifact_margin_x)
    x2 = min(width, int(np.ceil((xs.max() + 1) * sx)) - artifact_margin_x)
    y1 = max(0, int(np.floor(ys.min() * sy)) + artifact_margin_y)
    y2 = min(height, int(np.ceil((ys.max() + 1) * sy)) - artifact_margin_y)

    if x2 - x1 < width * 0.5 or y2 - y1 < height * 0.5:
        print('[Crop] auto-crop is very small; relaxing margin.')
        x1 = max(0, x1 - margin_x)
        x2 = min(width, x2 + margin_x)
        y1 = max(0, y1 - margin_y)
        y2 = min(height, y2 + margin_y)

    safe_w, safe_h = rotated_rect_with_max_area(width, height, np.deg2rad(float(np.max(np.abs(angles)))))
    current_w = x2 - x1
    current_h = y2 - y1
    if safe_w < current_w or safe_h < current_h:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        crop_w = min(current_w, safe_w)
        crop_h = min(current_h, safe_h)
        x1 = max(0, int(round(center_x - crop_w / 2.0)))
        x2 = min(width, x1 + int(crop_w))
        y1 = max(0, int(round(center_y - crop_h / 2.0)))
        y2 = min(height, y1 + int(crop_h))
        x1 = max(0, x2 - int(crop_w))
        y1 = max(0, y2 - int(crop_h))

    print(f'[Crop] box x=({x1}, {x2}) y=({y1}, {y2})')
    return x1, x2, y1, y2


def rotated_rect_with_max_area(width, height, angle_rad):
    if width <= 0 or height <= 0:
        return 0, 0

    angle_rad = abs(angle_rad)
    if angle_rad < 1e-8:
        return int(width), int(height)

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)
    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))

    if side_short <= 2.0 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        if width_is_longer:
            crop_w = x / sin_a
            crop_h = x / cos_a
        else:
            crop_w = x / cos_a
            crop_h = x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        crop_w = (width * cos_a - height * sin_a) / cos_2a
        crop_h = (height * cos_a - width * sin_a) / cos_2a

    return max(1, min(int(crop_w), width)), max(1, min(int(crop_h), height))


def fit_crop_box_to_aspect(x1, x2, y1, y2, target_width, target_height):
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    target_aspect = target_width / float(target_height)
    crop_aspect = crop_w / float(crop_h)

    if crop_aspect > target_aspect:
        new_w = int(round(crop_h * target_aspect))
        center_x = (x1 + x2) / 2.0
        x1 = int(round(center_x - new_w / 2.0))
        x2 = x1 + new_w
    elif crop_aspect < target_aspect:
        new_h = int(round(crop_w / target_aspect))
        center_y = (y1 + y2) / 2.0
        y1 = int(round(center_y - new_h / 2.0))
        y2 = y1 + new_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > target_width:
        x1 -= x2 - target_width
        x2 = target_width
    if y2 > target_height:
        y1 -= y2 - target_height
        y2 = target_height

    x1 = max(0, min(x1, target_width - 2))
    y1 = max(0, min(y1, target_height - 2))
    x2 = max(x1 + 2, min(x2, target_width))
    y2 = max(y1 + 2, min(y2, target_height))
    return x1, x2, y1, y2


def make_even_size(width, height):
    width = max(2, int(width) - (int(width) % 2))
    height = max(2, int(height) - (int(height) % 2))
    return width, height


def crop_frame_by_box(frame, crop_box):
    x1, x2, y1, y2 = crop_box
    return frame[y1:y2, x1:x2]


def render_upright_video(video_path, output_path, angles, info, auto_crop=True,
                         crop_sample_step=CROP_SAMPLE_STEP, crop_erode_iter=CROP_ERODE_ITER,
                         crop_eval_scale=CROP_EVAL_SCALE, crop_border_margin=CROP_BORDER_MARGIN,
                         crop_progress_callback=None, render_progress_callback=None):
    width = int(info['width'])
    height = int(info['height'])
    fps = float(info['fps'])
    n_frames = int(info['n_frames'])

    if auto_crop:
        x1, x2, y1, y2 = compute_auto_crop_box(
            angles,
            width,
            height,
            sample_step=crop_sample_step,
            erode_iter=crop_erode_iter,
            eval_scale=crop_eval_scale,
            border_margin=crop_border_margin,
            progress_callback=crop_progress_callback,
        )
    else:
        ratio = crop_border_margin if crop_border_margin > 0 else 0.12
        mx = int(width * ratio)
        my = int(height * ratio)
        x1, x2, y1, y2 = mx, width - mx, my, height - my

    out_w, out_h = width, height
    crop_w = x2 - x1
    crop_h = y2 - y1
    print(f'[Crop] render crop size: {crop_w}x{crop_h}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot reopen video: {video_path}')

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open output video writer: {output_path}')

    try:
        for frame_idx in tqdm(range(n_frames), desc='upright render', unit='frame'):
            ret, frame = cap.read()
            if not ret:
                break

            angle = float(angles[frame_idx])
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
            corrected = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            cropped = corrected[y1:y2, x1:x2]
            # 렌더 후 외곽 보간 흔적을 숨기는 추가 크롭입니다. 값을 키우면 더 확대됩니다.
            guard_x = max(2, int(round(cropped.shape[1] * 0.035)))
            guard_y = max(2, int(round(cropped.shape[0] * 0.035)))
            if cropped.shape[1] > guard_x * 2 and cropped.shape[0] > guard_y * 2:
                cropped = cropped[guard_y:cropped.shape[0] - guard_y, guard_x:cropped.shape[1] - guard_x]
            guard_h, guard_w = cropped.shape[:2]
            gx1, gx2, gy1, gy2 = fit_crop_box_to_aspect(0, guard_w, 0, guard_h, out_w, out_h)
            cropped = cropped[gy1:gy2, gx1:gx2]
            corrected = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            if corrected.shape[1] != out_w or corrected.shape[0] != out_h:
                corrected = cv2.resize(corrected, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            out.write(corrected)
            if render_progress_callback is not None:
                rendered = frame_idx + 1
                percent = int(round(rendered / max(n_frames, 1) * 100))
                render_progress_callback(min(percent, 100), f'Rendering upright result... {rendered}/{n_frames} frames')
    finally:
        cap.release()
        out.release()

    print(f'[Done] saved: {output_path}')
