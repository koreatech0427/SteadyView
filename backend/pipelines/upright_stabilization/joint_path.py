import cv2
import numpy as np
from scipy.optimize import least_squares

from config import LAMBDA, TAU


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


def rotation_homography(angle_deg, width, height):
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -float(angle_deg), 1.0)
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix
    return homography


def build_joint_paths(camera_path, angles_deg, video_info):
    h, w = video_info['height'], video_info['width']
    n_frames = camera_path.shape[0]
    rot_mats = np.stack(
        [rotation_homography(angles_deg[t], w, h) for t in range(n_frames)],
        axis=0,
    )
    r0_inv = np.linalg.inv(rot_mats[0])
    joint_camera_path = np.zeros_like(camera_path)

    for t in range(n_frames):
        joint_camera_path[t] = np.matmul(np.matmul(rot_mats[t], camera_path[t]), r0_inv)
        joint_camera_path[t] /= (joint_camera_path[t, ..., 2:3, 2:3] + 1e-10)

    return rot_mats, joint_camera_path

