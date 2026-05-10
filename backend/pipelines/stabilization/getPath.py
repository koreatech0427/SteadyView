# python_src/getPath.py
import numpy as np
from new_warping import new_warping

def get_path(mesh_size, tracks_data, video_info):
    n_frames = len(tracks_data) + 1
    H, W = video_info['height'], video_info['width']
    qH, qW = H // mesh_size, W // mesh_size
    alpha = 500.0

    # (Frame, Row, Col, 3, 3) 텐서 초기화
    path = np.zeros((n_frames, mesh_size, mesh_size, 3, 3))

    # 첫 프레임은 움직임이 없는 단위 행렬(Identity)로 초기화합니다.
    path[0, :, :] = np.tile(np.eye(3), (mesh_size, mesh_size, 1, 1))

    I_mat = np.tile(np.eye(3), (mesh_size, mesh_size, 1, 1))

    # 감쇠 계수: 1.0에 가까울수록 흔들림 보정이 강해지지만 왜곡 위험 증가
    gamma = 0.98

    for f in range(1, n_frames):
        pa, pb = tracks_data[f - 1]

        if pa is None or len(pa) < 4:
            homos = I_mat.copy()
        else:
            pa = pa.astype(np.float32)
            pb = pb.astype(np.float32)
            homos = new_warping(pa, pb, H, W, qH, qW, alpha)

        # 센서 데이터 연산 없이 순수 비전 추적 결과(homos)를 그대로 사용
        temp = homos.copy()
        temp /= (temp[..., 2:3, 2:3] + 1e-10)

        # 궤적 누적
        new_p = np.matmul(temp, path[f - 1])

        # --- [핵심 해결책: Identity Blending] ---
        new_p = (new_p * gamma) + (I_mat * (1.0 - gamma))
        new_p /= (new_p[..., 2:3, 2:3] + 1e-10)

        # 유효성 검사 매트릭스 생성
        dets = np.linalg.det(new_p)
        invalid_mask = np.isnan(new_p).any(axis=(2, 3)) | (np.abs(dets) < 1e-9)

        # 유효하지 않은 격자는 이전 프레임 값 유지, 나머지는 새 값 업데이트
        path[f] = np.where(invalid_mask[..., None, None], path[f - 1], new_p)

        print(f"궤적 계산 진행 중: {f}/{n_frames - 1}", end='\r')

    print("\n궤적 계산 완료!")
    return path