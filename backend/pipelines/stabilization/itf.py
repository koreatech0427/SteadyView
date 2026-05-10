import cv2
import numpy as np
import os


def calculate_itf(video_path, crop_ratio=0.2):
    """
    영상 가장자리의 검은 여백(Black Border)을 제외하고,
    중앙부 실제 콘텐츠의 프레임 간 PSNR 평균(ITF)을 계산합니다.
    """
    if not os.path.exists(video_path):
        print(f"❌ 파일을 찾을 수 없습니다: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 테두리 노이즈를 피하기 위해 상하좌우 20%씩 제외한 중앙 60% 영역만 타겟팅
    y1, y2 = int(height * crop_ratio), int(height * (1 - crop_ratio))
    x1, x2 = int(width * crop_ratio), int(width * (1 - crop_ratio))

    psnr_values = []
    ret, prev_frame = cap.read()
    if not ret:
        return 0.0

    # 첫 프레임: 중앙 크롭 후 그레이스케일 변환
    prev_gray = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    print(f"🎬 분석 중: {os.path.basename(video_path)}")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 현재 프레임: 중앙 크롭 후 그레이스케일 변환
        curr_gray = cv2.cvtColor(curr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        # MSE(Mean Squared Error) 계산: 두 이미지의 픽셀 차이
        mse = np.mean((prev_gray.astype(np.float32) - curr_gray.astype(np.float32)) ** 2)

        # MSE가 0보다 클 때만 PSNR 계산 (완전히 똑같은 프레임이면 무한대가 되므로 제외)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
            psnr_values.append(psnr)

        prev_gray = curr_gray

    cap.release()

    # 평균 PSNR = ITF
    return np.mean(psnr_values) if psnr_values else 0.0


# ==========================================
# 🚀 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 방금 은준님이 파이프라인으로 돌리셨던 영상 경로 그대로 넣었습니다.
    original_video = r"C:\Users\korea\Desktop\Stabilization\Stabilization_compare\datasets\input\004_input.mp4"
    stabilized_video = r"C:\Users\korea\Desktop\Stabilization\Stabilization_compare\datasets\output\output_36.mp4"

    print("📊 ITF(Inter-frame Transformation Fidelity) 점수 측정 시작...\n")

    itf_org = calculate_itf(original_video, crop_ratio=0.2)
    itf_stab = calculate_itf(stabilized_video, crop_ratio=0.2)

    print("\n" + "=" * 40)
    print(f"🏆 최종 ITF 평가 결과 (단위: dB)")
    print("-" * 40)
    if itf_org: print(f"🔹 원본 영상 (Original)  : {itf_org:.2f} dB")
    if itf_stab: print(f"🔥 보정 영상 (SteadyView): {itf_stab:.2f} dB")
    print("-" * 40)

    if itf_org and itf_stab:
        print(f"✨ 성능 향상 폭: +{(itf_stab - itf_org):.2f} dB")
    print("=" * 40)