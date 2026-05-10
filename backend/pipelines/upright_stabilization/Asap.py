import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


class AsapPy:
    def __init__(self, H, W, qH, qW, alpha):
        self.imgHeight, self.imgWidth, self.alpha = H, W, alpha
        self.quadHeight, self.quadWidth = qH, qW

        # Mesh 생성 로직
        x_set, y_set = [0.0], [0.0]
        curr_x, curr_y = 0.0, 0.0
        while W - curr_x > 0.5 * qW:
            curr_x += qW
            x_set.append(curr_x)
        while H - curr_y > 0.5 * qH:
            curr_y += qH
            y_set.append(curr_y)

        self.meshHeight, self.meshWidth = len(y_set), len(x_set)
        self.num_vertices = self.meshHeight * self.meshWidth

        # 원본 격자 정점 좌표
        self.src_x = np.tile(x_set, (self.meshHeight, 1))
        self.src_y = np.tile(y_set, (self.meshWidth, 1)).T
        self.data_weights, self.data_quad_idx, self.data_pts_dst = [], [], []

    def add_control_points(self, p1, p2):
        for i in range(len(p1)):
            # [수정된 부분] 화면 밖으로 나간 점(음수 좌표)은 무시하도록 예외 처리 추가
            if p1[i, 0] < 0 or p1[i, 1] < 0: continue

            gi = int(p1[i, 1] // self.quadHeight)
            gj = int(p1[i, 0] // self.quadWidth)

            # 격자 범위를 벗어나는 경우도 무시
            if gi >= self.meshHeight - 1 or gj >= self.meshWidth - 1: continue

            x1, x2 = gj * self.quadWidth, (gj + 1) * self.quadWidth
            y1, y2 = gi * self.quadHeight, (gi + 1) * self.quadHeight

            # 분모가 0이 되는 것을 방지하기 위해 고정값 사용 (또는 1e-6 추가)
            denom = self.quadWidth * self.quadHeight

            w00 = (x2 - p1[i, 0]) * (y2 - p1[i, 1]) / denom
            w01 = (p1[i, 0] - x1) * (y2 - p1[i, 1]) / denom
            w10 = (x2 - p1[i, 0]) * (p1[i, 1] - y1) / denom
            w11 = (p1[i, 0] - x1) * (p1[i, 1] - y1) / denom

            self.data_weights.append([w00, w01, w10, w11])
            self.data_quad_idx.append((gi, gj))
            self.data_pts_dst.append(p2[i])

    def solve(self):
        # 데이터가 없으면 None 반환
        if not self.data_weights:
            return None

        rows, cols, data, b = [], [], [], []
        curr_row = 0

        # 1. Data Term (특징점 위치 제약)
        for i in range(len(self.data_weights)):
            w = self.data_weights[i]
            gi, gj = self.data_quad_idx[i]
            dst_pt = self.data_pts_dst[i]

            v_idx = [
                gi * self.meshWidth + gj, gi * self.meshWidth + gj + 1,
                (gi + 1) * self.meshWidth + gj, (gi + 1) * self.meshWidth + gj + 1
            ]

            for axis in range(2):
                for k in range(4):
                    rows.append(curr_row)
                    cols.append(v_idx[k] * 2 + axis)
                    data.append(w[k])
                b.append(dst_pt[axis])
                curr_row += 1

        # 2. Smoothness Term (격자 모양 유지)
        alpha = self.alpha

        # (1) 가로 방향 연결 (Horizontal)
        for i in range(self.meshHeight):
            for j in range(self.meshWidth - 1):
                v0 = (i * self.meshWidth + j) * 2
                v1 = (i * self.meshWidth + (j + 1)) * 2

                # X축: v0(x) - v1(x) = -width
                rows.extend([curr_row, curr_row])
                cols.extend([v0, v1])
                data.extend([alpha, -alpha])
                b.append(alpha * (-self.quadWidth))
                curr_row += 1

                # Y축: v0(y) - v1(y) = 0
                rows.extend([curr_row, curr_row])
                cols.extend([v0 + 1, v1 + 1])
                data.extend([alpha, -alpha])
                b.append(0)
                curr_row += 1

        # (2) 세로 방향 연결 (Vertical)
        for i in range(self.meshHeight - 1):
            for j in range(self.meshWidth):
                v0 = (i * self.meshWidth + j) * 2
                v1 = ((i + 1) * self.meshWidth + j) * 2

                # X축: v0(x) - v1(x) = 0
                rows.extend([curr_row, curr_row])
                cols.extend([v0, v1])
                data.extend([alpha, -alpha])
                b.append(0)
                curr_row += 1

                # Y축: v0(y) - v1(y) = -height
                rows.extend([curr_row, curr_row])
                cols.extend([v0 + 1, v1 + 1])
                data.extend([alpha, -alpha])
                b.append(alpha * (-self.quadHeight))
                curr_row += 1

        if curr_row == 0: return None

        # 희소 행렬 풀이
        A = coo_matrix((data, (rows, cols)), shape=(curr_row, self.num_vertices * 2)).tocsr()
        res = lsqr(A, b)

        # [핵심] 결과를 (2, N) 형태로 전치(.T)하여 반환
        return res[0].reshape(-1, 2).T