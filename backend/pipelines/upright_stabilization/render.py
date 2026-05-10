import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm


def compute_joint_source_map(rot_h, joint_camera_t, joint_stable_t, out_width, out_height,
                             full_width=None, full_height=None, crop_x1=0, crop_y1=0,
                             coord_width=None, coord_height=None):
    mesh_rows, mesh_cols = joint_camera_t.shape[:2]
    frame_w = full_width if full_width is not None else out_width
    frame_h = full_height if full_height is not None else out_height
    coord_w = coord_width if coord_width is not None else frame_w
    coord_h = coord_height if coord_height is not None else frame_h
    cell_w = frame_w / float(mesh_cols)
    cell_h = frame_h / float(mesh_rows)

    coord_x = np.linspace(crop_x1, crop_x1 + coord_w, mesh_cols + 1, dtype=np.float32)
    coord_y = np.linspace(crop_y1, crop_y1 + coord_h, mesh_rows + 1, dtype=np.float32)
    src_x, src_y = np.meshgrid(coord_x, coord_y)

    dst_v_x = np.zeros_like(src_x, dtype=np.float32)
    dst_v_y = np.zeros_like(src_y, dtype=np.float32)
    rot_inv = np.linalg.inv(rot_h)

    for i in range(mesh_rows + 1):
        for j in range(mesh_cols + 1):
            px = float(src_x[i, j])
            py = float(src_y[i, j])
            ri = int(np.clip(py / max(cell_h, 1e-10), 0, mesh_rows - 1))
            ci = int(np.clip(px / max(cell_w, 1e-10), 0, mesh_cols - 1))

            stable_inv = np.linalg.pinv(joint_stable_t[ri, ci])
            joint_inv = rot_inv @ joint_camera_t[ri, ci] @ stable_inv

            p = np.array([px, py, 1.0], dtype=np.float64)
            p_src = joint_inv @ p
            z = p_src[2] if abs(p_src[2]) > 1e-10 else 1e-10

            dst_v_x[i, j] = float(p_src[0] / z)
            dst_v_y[i, j] = float(p_src[1] / z)

    map_x = cv2.resize(dst_v_x, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(dst_v_y, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def compute_auto_crop_box(rot_mats, joint_camera_path, joint_stable_path, video_info,
                          sample_step=1, erode_iter=5, eval_scale=0.5, border_margin=0.02):
    h, w = video_info['height'], video_info['width']
    eval_w = max(128, int(round(w * eval_scale)))
    eval_h = max(128, int(round(h * eval_scale)))
    common_valid = np.ones((eval_h, eval_w), dtype=bool)
    n_frames = joint_camera_path.shape[0]

    frame_indices = list(range(0, n_frames, max(1, sample_step)))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    # Cubic interpolation samples a neighborhood around each source coordinate.
    # Keep a stronger source margin so border pixels are not stretched into streaks.
    source_margin = 8.0
    src_x_min, src_x_max = source_margin, w - 1.0 - source_margin
    src_y_min, src_y_max = source_margin, h - 1.0 - source_margin

    for t in tqdm(frame_indices, desc='auto-crop mask', unit='frame'):
        map_x, map_y = compute_joint_source_map(
            rot_mats[t], joint_camera_path[t], joint_stable_path[t],
            out_width=eval_w, out_height=eval_h,
            full_width=w, full_height=h,
            coord_width=w, coord_height=h,
        )
        valid = (
            np.isfinite(map_x) & np.isfinite(map_y) &
            (map_x >= src_x_min) & (map_x <= src_x_max) &
            (map_y >= src_y_min) & (map_y <= src_y_max)
        )
        if erode_iter > 0:
            valid = binary_erosion(valid, iterations=erode_iter)
        common_valid &= valid

    ys, xs = np.where(common_valid)
    if len(xs) == 0 or len(ys) == 0:
        print('[Crop] auto-crop failed; using full frame.')
        return 0, w, 0, h

    sx, sy = w / float(eval_w), h / float(eval_h)
    margin_x = int(np.ceil(w * border_margin))
    margin_y = int(np.ceil(h * border_margin))
    artifact_margin_x = max(margin_x, int(np.ceil(w * 0.10)))
    artifact_margin_y = max(margin_y, int(np.ceil(h * 0.10)))

    x1 = max(0, int(np.floor(xs.min() * sx)) + artifact_margin_x)
    x2 = min(w, int(np.ceil((xs.max() + 1) * sx)) - artifact_margin_x)
    y1 = max(0, int(np.floor(ys.min() * sy)) + artifact_margin_y)
    y2 = min(h, int(np.ceil((ys.max() + 1) * sy)) - artifact_margin_y)

    if x2 - x1 < w * 0.5 or y2 - y1 < h * 0.5:
        print('[Crop] auto-crop is very small; relaxing margin.')
        x1 = max(0, x1 - margin_x)
        x2 = min(w, x2 + margin_x)
        y1 = max(0, y1 - margin_y)
        y2 = min(h, y2 + margin_y)

    print(f'[Crop] box x=({x1}, {x2}) y=({y1}, {y2})')
    return x1, x2, y1, y2


def inset_crop_box(x1, x2, y1, y2, inset_ratio=0.04):
    crop_w = x2 - x1
    crop_h = y2 - y1
    inset_x = int(round(crop_w * inset_ratio))
    inset_y = int(round(crop_h * inset_ratio))
    return x1 + inset_x, x2 - inset_x, y1 + inset_y, y2 - inset_y


def render_joint_video(video_path, output_path, rot_mats, joint_camera_path, joint_stable_path,
                       video_info, auto_crop=True, crop_sample_step=1, crop_erode_iter=5,
                       crop_eval_scale=0.5, crop_border_margin=0.02):
    h, w = video_info['height'], video_info['width']
    fps = video_info['fps']
    n_frames = joint_camera_path.shape[0]

    if auto_crop:
        x1, x2, y1, y2 = compute_auto_crop_box(
            rot_mats, joint_camera_path, joint_stable_path, video_info,
            sample_step=crop_sample_step,
            erode_iter=crop_erode_iter,
            eval_scale=crop_eval_scale,
            border_margin=crop_border_margin,
        )
    else:
        ratio = crop_border_margin if crop_border_margin > 0 else 0.12
        mx = int(w * ratio)
        my = int(h * ratio)
        x1, x2, y1, y2 = mx, w - mx, my, h - my

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot reopen video: {video_path}')

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    crop_w = x2 - x1
    crop_h = y2 - y1
    print(f'[Crop] render crop size: {crop_w}x{crop_h}')

    for t in tqdm(range(n_frames), desc='joint render', unit='frame'):
        ret, frame = cap.read()
        if not ret:
            break

        map_x, map_y = compute_joint_source_map(
            rot_mats[t], joint_camera_path[t], joint_stable_path[t],
            out_width=crop_w,
            out_height=crop_h,
            full_width=w,
            full_height=h,
            crop_x1=x1,
            crop_y1=y1,
            coord_width=crop_w,
            coord_height=crop_h,
        )
        cropped = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        guard_x = max(2, int(round(cropped.shape[1] * 0.035)))
        guard_y = max(2, int(round(cropped.shape[0] * 0.035)))
        if cropped.shape[1] > guard_x * 2 and cropped.shape[0] > guard_y * 2:
            cropped = cropped[guard_y:cropped.shape[0] - guard_y, guard_x:cropped.shape[1] - guard_x]
        final_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        out.write(final_frame)

    cap.release()
    out.release()
    print(f'[Done] saved: {output_path}')
