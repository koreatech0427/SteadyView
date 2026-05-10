# main.py
import os
import sys
import torch
import numpy as np
import cv2

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'Stabilization', 'python_src'))

# 모듈 임포트
from tracker import get_tracks
from getPath import get_path
from smoothPath import smooth_path
from stabilizer import render_combined_video

def get_actual_frame_count(video_path):
    """실제 비디오 프레임 수 확인"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def main():
    print("🚀 Steady View 통합 파이프라인 시작! (Stabilization Only)")

    # 1. 파일 및 경로 설정
    video_path = r"C:\Users\korea\Desktop\Stabilization\Stabilization_compare\datasets\input\006_input.mp4"
    output_path = r"C:\Users\korea\Desktop\Stabilization\Stabilization_compare\datasets\output\006_output.mp4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_frames = get_actual_frame_count(video_path)

    print(f"📹 비디오 프레임 수: {n_frames}")
    print(f"🖥️  사용 디바이스: {device}")

    # 2. 특징점 추적 및 흔들림 궤적 계산
    mesh_size = 16
    demand = 1024

    print("🔍 1단계: 특징점 추적 및 흔들림 궤적 계산...")
    matched_pairs, info = get_tracks(video_path, mesh_size, demand, n_frames)

    if not matched_pairs:
        print("❌ 특징점 추적 실패. 종료합니다.")
        return

    # 추적 프레임 수 재조정
    actual_n = len(matched_pairs)
    if actual_n != n_frames - 1:
        print(f"⚠️  추적 프레임 수 불일치: 예상 {n_frames - 1}, 실제 {actual_n} → 조정")
        n_frames = actual_n + 1

    # 원본 해상도 좌표 복원
    scale_inv = 1.0 / info['scale']
    rescaled_pairs = [
        (pa.astype(np.float32) * scale_inv, pb.astype(np.float32) * scale_inv)
        for pa, pb in matched_pairs
    ]

    # 3. 궤적 최적화 및 스무딩
    sigma = min(16, n_frames // 5)
    print(f"📐 스무딩 sigma: {sigma} (프레임 수: {n_frames})")

    # 순수 궤적 계산 (theta 제외)
    camera_path = get_path(mesh_size, rescaled_pairs, info)

    # 궤적 스무딩 실행
    smoothed = smooth_path(camera_path, sigma=sigma)

    # 4. 최종 렌더링
    print("🎥 2단계: 프레임 렌더링 시작...")
    render_combined_video(
        video_path, output_path,
        camera_path, smoothed,
        info, None, device
    )

    print("🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()