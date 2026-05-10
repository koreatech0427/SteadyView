# python_src/getPath.py
import numpy as np

from new_warping import new_warping


def get_path(mesh_size, tracks_data, video_info, theta_input):
    n_frames = len(tracks_data) + 1
    H, W = video_info['height'], video_info['width']
    qH, qW = H // mesh_size, W // mesh_size
    alpha = 500.0

    # (Frame, Row, Col, 3, 3) 텐서 초기화
    path = np.zeros((n_frames, mesh_size, mesh_size, 3, 3))

    moveC = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])
    moveTL = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]])

    # ----- [센서 데이터 관련 부분 주석 처리] -----
    # angle1 = theta_input[0]
    # R1 = np.array([
    #     [np.cos(angle1), -np.sin(angle1), 0],
    #     [np.sin(angle1), np.cos(angle1), 0],
    #     [0, 0, 1]
    # ])
    # R1 = moveTL @ R1 @ moveC
    # path[0, :, :] = R1

    # 센서 데이터가 없으므로 첫 프레임은 움직임이 없는 단위 행렬(Identity)로 초기화합니다.
    path[0, :, :] = np.tile(np.eye(3), (mesh_size, mesh_size, 1, 1))
    # ---------------------------------------------

    I_mat = np.tile(np.eye(3), (mesh_size, mesh_size, 1, 1))

    # 감쇠 계수: 1.0에 가까울수록 흔들림 보정이 강해지지만 왜곡 위험 증가
    # 0.98 정도면 500프레임 이상에서도 궤적이 폭발하지 않도록 꽉 잡아줍니다.
    gamma = 0.98

    for f in range(1, n_frames):
        pa, pb = tracks_data[f - 1]

        if pa is None or len(pa) < 4:
            homos = I_mat.copy()
        else:
            pa = pa.astype(np.float32)
            pb = pb.astype(np.float32)
            homos = new_warping(pa, pb, H, W, qH, qW, alpha)

        # ----- [센서 데이터 관련 부분 주석 처리] -----
        # R0_val, R1_val = theta_input[f - 1], theta_input[f]
        # M0 = np.array([[np.cos(R0_val), np.sin(R0_val), 0], [-np.sin(R0_val), np.cos(R0_val), 0], [0, 0, 1]])
        # M1 = np.array([[np.cos(R1_val), -np.sin(R1_val), 0], [np.sin(R1_val), np.cos(R1_val), 0], [0, 0, 1]])
        # M0, M1 = moveTL @ M0 @ moveC, moveTL @ M1 @ moveC
        #
        # # M1 @ homos @ M0 연산을 텐서 브로드캐스팅으로 일괄 처리
        # temp = np.matmul(np.matmul(M1, homos), M0)

        # 센서 데이터(M0, M1) 연산을 생략하고 순수 비전 추적 결과(homos)를 그대로 사용합니다.
        temp = homos.copy()
        # ---------------------------------------------

        temp /= (temp[..., 2:3, 2:3] + 1e-10)

        # 1. 기존처럼 궤적 누적
        new_p = np.matmul(temp, path[f - 1])

        # --- [핵심 해결책: Identity Blending] ---
        # 누적된 행렬을 그대로 쓰지 않고, 매 프레임 단위 행렬(I_mat) 쪽으로 2%씩 당겨줍니다.
        # 이렇게 하면 스케일 발산, 이동량 폭주, 원근 붕괴가 수학적으로 모두 억제됩니다.
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