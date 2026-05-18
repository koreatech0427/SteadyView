# python_src/tracker.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

_LOFTR_CACHE = {}


def _get_loftr_matcher(device):
    cache_key = str(device)
    matcher = _LOFTR_CACHE.get(cache_key)
    if matcher is None:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        matcher = LoFTR(pretrained='outdoor').to(device).eval()
        _LOFTR_CACHE[cache_key] = matcher
    return matcher


def get_tracks(video_path, mesh_size, demand, n_frames, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return None, "영상 로드 실패"

    h, w = first_frame.shape[:2]

    # LoFTR 모델이 요구하는 텐서 규격('8의 배수')을 맞추고,
    # RTX 3060 VRAM에 맞게 안정적인 해상도로 세팅합니다.
    user_scale = 540.0 / h
    new_h = (int(540.0) // 8) * 8
    scale = new_h / h
    new_w = (int(w * scale) // 8) * 8

    # PyTorch GPU 및 사전 학습된 LoFTR 모델 로드 (최초 실행 시 가중치 자동 다운로드)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = _get_loftr_matcher(device)

    def preprocess(frame):
        # 1. 크기 조절 및 흑백 변환
        gray = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2GRAY)
        # 2. OpenCV Numpy 배열 -> PyTorch Tensor 브릿지 (0~1 정규화)
        tensor = torch.from_numpy(gray).float() / 255.0
        # 3. [Batch, Channel, Height, Width] 차원 추가 후 GPU로 전송
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
        return tensor

    prev_tensor = preprocess(first_frame)
    all_matched_pairs = []

    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret: break

        curr_tensor = preprocess(frame)

        # ✨ [딥러닝 추론] 두 이미지를 넣고 특징점 좌표 쌍을 뽑아냅니다.
        with torch.inference_mode():
            input_dict = {"image0": prev_tensor, "image1": curr_tensor}
            correspondences = matcher(input_dict)

        # PyTorch Tensor -> 기존 코드(RANSAC, ASAP)가 호환되는 OpenCV(NumPy)로 복원
        pa = correspondences['keypoints0'].cpu().numpy()
        pb = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()

        # 신뢰도 0.8 이상의 확실한 점(Inliers)만 필터링하여 RANSAC 부담 완화
        good_mask = confidence > 0.8
        pa = pa[good_mask]
        pb = pb[good_mask]

        # 기존 steady_view_main.py의 스케일 복원 로직과 호환되도록 미세 좌표 보정
        pa[:, 0] = pa[:, 0] * (w * user_scale / new_w)
        pa[:, 1] = pa[:, 1] * (h * user_scale / new_h)
        pb[:, 0] = pb[:, 0] * (w * user_scale / new_w)
        pb[:, 1] = pb[:, 1] * (h * user_scale / new_h)

        all_matched_pairs.append((pa, pb))
        prev_tensor = curr_tensor
        if progress_callback is not None and n_frames > 1:
            percent = int(round(i / (n_frames - 1) * 100))
            progress_callback(min(percent, 100), f"특징점 추적 중... {i}/{n_frames - 1}프레임")

        print(f"✨ LoFTR 딥러닝 특징점 추적 중... {i}/{n_frames} (찾은 매칭 수: {len(pa)}개)", end='\r')

    cap.release()
    return all_matched_pairs, {'width': w, 'height': h, 'scale': user_scale}

# (참고) 기존에 있던 get_more_points 함수는 LoFTR가 화면 전체에서
# 알아서 수천 개의 점을 촘촘하게 찾아주므로 더 이상 필요 없어서 깔끔하게 지웠습니다!
