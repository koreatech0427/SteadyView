import cv2
import numpy as np


def get_rotation_matrix(angle_degrees, w, h):
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_degrees, 1.0)
    return np.vstack([matrix, [0, 0, 1]])


def render_combined_video(
    video_path,
    output_path,
    camera_path,
    smooth_path,
    info,
    upright_model,
    device,
    crop_ratio=0.12,
    alpha=0.04,
    progress_callback=None,
    cancel_callback=None,
):
    def check_cancel():
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("JobCancelled")

    check_cancel()
    cap = cv2.VideoCapture(video_path)
    n_frames, mesh_rows, mesh_cols, _, _ = camera_path.shape
    h, w = info["height"], info["width"]
    q_h, q_w = h // mesh_rows, w // mesh_cols
    smooth_inv_path = np.linalg.pinv(smooth_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("[Crop] Calculating adaptive crop bounds...")
    max_left, max_top = 0, 0
    min_right, min_bottom = w, h

    for frame_idx in range(n_frames):
        check_cancel()
        corners = [
            (0, 0, 0, 0),
            (w, 0, mesh_cols - 1, 0),
            (0, h, 0, mesh_rows - 1),
            (w, h, mesh_cols - 1, mesh_rows - 1),
        ]

        for cx, cy, grid_col, grid_row in corners:
            warp_h = np.matmul(camera_path[frame_idx, grid_row, grid_col], smooth_inv_path[frame_idx, grid_row, grid_col])
            forward_h = np.linalg.pinv(warp_h)
            point = np.array([cx, cy, 1.0])
            mapped = np.matmul(forward_h, point)

            px = mapped[0] / (mapped[2] + 1e-10)
            py = mapped[1] / (mapped[2] + 1e-10)

            if cx == 0:
                max_left = max(max_left, px)
            if cx == w:
                min_right = min(min_right, px)
            if cy == 0:
                max_top = max(max_top, py)
            if cy == h:
                min_bottom = min(min_bottom, py)

    buffer = 30
    x1 = min(int(max_left + buffer), int(w * 0.30))
    y1 = min(int(max_top + buffer), int(h * 0.30))
    x2 = max(int(min_right - buffer), int(w * 0.70))
    y2 = max(int(min_bottom - buffer), int(h * 0.70))
    print(f"[Crop] box x=({x1}, {x2}) y=({y1}, {y2})")

    src_y, src_x = np.mgrid[0:mesh_rows + 1, 0:mesh_cols + 1]
    src_y = (src_y * q_h).astype(np.float32)
    src_x = (src_x * q_w).astype(np.float32)

    print("[Render] Final rendering started. (Stabilization mode)")
    for frame_idx in range(n_frames):
        check_cancel()
        ret, frame = cap.read()
        if not ret:
            break

        predicted_angle = 0.0
        if upright_model is not None:
            predicted_angle = upright_model.predict_angle(frame)

        r_upright = get_rotation_matrix(predicted_angle, w, h)
        dst_v_x = np.zeros_like(src_x)
        dst_v_y = np.zeros_like(src_y)

        for row in range(mesh_rows + 1):
            for col in range(mesh_cols + 1):
                grid_row = min(row, mesh_rows - 1)
                grid_col = min(col, mesh_cols - 1)

                warp_h = np.matmul(camera_path[frame_idx, grid_row, grid_col], smooth_inv_path[frame_idx, grid_row, grid_col])
                combined_h = np.matmul(r_upright, warp_h)

                point = np.array([src_x[row, col], src_y[row, col], 1.0])
                mapped = np.matmul(combined_h, point)

                dst_v_x[row, col] = mapped[0] / (mapped[2] + 1e-10)
                dst_v_y[row, col] = mapped[1] / (mapped[2] + 1e-10)

        map_x = cv2.resize(dst_v_x, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(dst_v_y, (w, h), interpolation=cv2.INTER_LINEAR)

        stabilized = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        final_crop = stabilized[y1:y2, x1:x2]
        final_frame = cv2.resize(final_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        out.write(final_frame)

        print(f"Progress: {frame_idx + 1}/{n_frames}", end="\r")
        if progress_callback is not None:
            rendered = frame_idx + 1
            percent = int(round(rendered / n_frames * 100))
            progress_callback(min(percent, 100), f"결과 영상 렌더링 중... {rendered}/{n_frames}프레임")

    cap.release()
    out.release()
