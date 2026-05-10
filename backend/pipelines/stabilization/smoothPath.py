# python_src/smoothPath.py
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

def smooth_path(camera_path, sigma=None):
    # 만약 sigma가 전달되지 않으면 자체적으로 궤적의 변화율을 보고 판단
    if sigma is None:
        diff = np.diff(camera_path, axis=0)
        motion_score = np.abs(diff).mean() * 100
        sigma = np.clip(10 + motion_score, 15, 40)
        print(f"✨ 내부 계산된 동적 Sigma: {sigma:.2f}")

    despiked_path = median_filter(camera_path, size=(5, 1, 1, 1, 1))

    # 2. 가우시안 필터 적용
    sigma_tuple = (sigma, 1.5, 1.5, 0, 0)
    smoothed_path = gaussian_filter(despiked_path, sigma=sigma_tuple, mode='nearest')

    # 정규화
    smoothed_path /= (smoothed_path[..., 2:3, 2:3] + 1e-10)

    return smoothed_path