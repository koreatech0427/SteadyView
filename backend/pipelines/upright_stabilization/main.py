import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
STA_UP_DIR = os.path.join(PROJECT_ROOT, 'sta_up')
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, STA_UP_DIR)

from analysis import analyze_video_shared
from config import LAMBDA, TAU
from joint_path import build_joint_paths, optimize_video_angles, truncate_angles
from render import render_joint_video
from smoothPath import smooth_path
from getPath import get_path
from upright_model import build_transform, load_upright_model


def run_pipeline(video_path, model_path, output_path,
                 mesh_size=16, demand=1024,
                 tau=TAU, lam=LAMBDA,
                 auto_crop=True,
                 crop_sample_step=2,
                 crop_erode_iter=5,
                 crop_eval_scale=0.5,
                 # 기본 안전 여백입니다. 실제 체감 크롭은 render.py의 AUTO_CROP_ARTIFACT_MARGIN_RATIO,
                 # RENDER_EXTRA_GUARD_RATIO가 더 크게 좌우합니다.
                 crop_border_margin=0.02,
                 progress_callback=None):
    def report(progress, message):
        if progress_callback is not None:
            progress_callback(int(progress), message)

    def map_progress(start, end):
        def mapped(percent, message):
            progress = start + (end - start) * (max(0, min(int(percent), 100)) / 100.0)
            report(round(progress), message)
        return mapped

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Start] device: {device}')
    report(1, '보정 모델을 준비하고 있습니다.')

    upright_model = load_upright_model(model_path, device)
    transform = build_transform()
    print(f'[Model] loaded: {model_path}')
    report(5, '보정 모델을 불러왔습니다.')

    print('\n[1/5] LoFTR tracking + upright angle analysis')
    theta_os, sigmas_sq, phis, rescaled_pairs, info = analyze_video_shared(
        video_path, upright_model, transform, device, mesh_size, demand,
        progress_callback=map_progress(6, 55),
    )
    n_frames = info['n_frames']
    print(f'[Video] frames: {n_frames}')
    report(56, '영상 분석이 완료되었습니다.')

    if len(phis) < max(n_frames - 1, 0):
        phis = phis + [0.0] * (n_frames - 1 - len(phis))
    else:
        phis = phis[:n_frames - 1]

    print('\n[2/5] Upright MAP optimization')
    report(58, '회전 보정값을 최적화하고 있습니다.')
    final_thetas = optimize_video_angles(theta_os, sigmas_sq, phis, lam=lam)
    truncated_thetas = truncate_angles(final_thetas, tau=tau)
    print(f'  raw angle:       {min(theta_os):+.2f} ~ {max(theta_os):+.2f}')
    print(f'  optimized angle: {min(final_thetas):+.2f} ~ {max(final_thetas):+.2f}')
    print(f'  truncated angle: {min(truncated_thetas):+.2f} ~ {max(truncated_thetas):+.2f}')

    print('\n[3/5] Camera path')
    report(62, '카메라 경로를 계산하고 있습니다.')
    camera_path = get_path(mesh_size, rescaled_pairs, info, truncated_thetas)
    sigma = min(16, max(3, n_frames // 5))
    print(f'[Path] smoothing sigma: {sigma}')

    print('\n[4/5] Joint path')
    report(68, '흔들림과 수평 보정 경로를 합성하고 있습니다.')
    rot_mats, joint_camera_path = build_joint_paths(camera_path, truncated_thetas, info)
    joint_stable_path = smooth_path(joint_camera_path, sigma=sigma)

    print('\n[5/5] Joint render')
    report(72, '결과 영상 렌더링을 준비하고 있습니다.')
    render_joint_video(
        video_path, output_path,
        rot_mats, joint_camera_path, joint_stable_path,
        info,
        auto_crop=auto_crop,
        crop_sample_step=crop_sample_step,
        crop_erode_iter=crop_erode_iter,
        crop_eval_scale=crop_eval_scale,
        crop_border_margin=crop_border_margin,
        crop_progress_callback=map_progress(72, 80),
        render_progress_callback=map_progress(80, 100),
    )
    report(100, '수평/흔들림 보정이 완료되었습니다.')

    corrections = np.abs(truncated_thetas)
    print('\n[Stats]')
    print(f'  mean correction: {corrections.mean():.3f}')
    print(f'  max correction:  {corrections.max():.3f}')
    print(f'  >1 degree:       {(corrections > 1.0).sum()} frames ({(corrections > 1.0).mean() * 100:.1f}%)')
    print(f'  >5 degree:       {(corrections > 5.0).sum()} frames ({(corrections > 5.0).mean() * 100:.1f}%)')


if __name__ == '__main__':
    video_path = r"C:\Users\korea\Desktop\Stabilization\Stabilization+Upright\datasets\input\006_input.mp4"
    model_path = r"C:\Users\korea\Desktop\Stabilization\Stabilization+Upright\upright\best_stage2_true_hybrid_real_best.pth"
    output_path = r"C:\Users\korea\Desktop\Stabilization\Stabilization+Upright\datasets\output\sta_up_joint_006_last.mp4"

    run_pipeline(
        video_path=video_path,
        model_path=model_path,
        output_path=output_path,
        mesh_size=16,
        demand=1024,
        auto_crop=True,
        crop_sample_step=2,
        crop_erode_iter=5,
        crop_eval_scale=0.5,
        crop_border_margin=0.02,
    )
