# python_src/new_warping.py
import cv2
import numpy as np
from Asap import AsapPy

def new_warping(pa, pb, H, W, qH, qW, alpha):
    preH, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if preH is None:
        preH = np.eye(3)

    if mask is not None:
        inliers = mask.ravel() == 1
        pa = pa[inliers]
        pb = pb[inliers]

    if len(pa) == 0:
        rows, cols = H // qH, W // qW
        return np.tile(preH, (rows, cols, 1, 1))

    inv_preH = np.linalg.pinv(preH)
    pb_h = np.column_stack([pb, np.ones(len(pb))])
    pb_warp_h = (inv_preH @ pb_h.T).T
    pb_warp = pb_warp_h[:, :2] / (pb_warp_h[:, 2:] + 1e-10)

    asap = AsapPy(H, W, qH, qW, alpha)
    asap.add_control_points(pa, pb_warp)

    new_vertices = asap.solve()
    rows, cols = H // qH, W // qW

    if new_vertices is None:
        return np.tile(preH, (rows, cols, 1, 1))

    homos = np.zeros((rows, cols, 3, 3))
    mesh_w = asap.meshWidth

    for i in range(rows):
        for j in range(cols):
            v_idx = [i * mesh_w + j, i * mesh_w + j + 1, (i + 1) * mesh_w + j, (i + 1) * mesh_w + j + 1]

            src_pts = np.array([
                [j * qW, i * qH], [(j + 1) * qW, i * qH],
                [j * qW, (i + 1) * qH], [(j + 1) * qW, (i + 1) * qH]
            ], dtype=np.float32)

            dst_pts = np.array([
                [new_vertices[0, v_idx[0]], new_vertices[1, v_idx[0]]],
                [new_vertices[0, v_idx[1]], new_vertices[1, v_idx[1]]],
                [new_vertices[0, v_idx[2]], new_vertices[1, v_idx[2]]],
                [new_vertices[0, v_idx[3]], new_vertices[1, v_idx[3]]]
            ], dtype=np.float32)

            try:
                H_local = cv2.getPerspectiveTransform(src_pts, dst_pts)
                final_H = preH @ H_local
                homos[i, j] = final_H / (final_H[2, 2] + 1e-10)
            except:
                homos[i, j] = preH

    return homos