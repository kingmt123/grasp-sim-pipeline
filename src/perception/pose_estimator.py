# src/perception/pose_estimator.py
# 位姿估计模块
# 核心方法：mask + depth → 3D 点云 → 点云质心（世界坐标）
# 逐像素反投影后取 3D 质心，避免 2D→3D 投影的透视偏差。

import numpy as np
import pybullet as p

from src.configs.config import (
    CAM_FAR,
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_NEAR,
    CAM_WIDTH,
)


def get_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    f_x = (width / 2.0) / np.tan(fov_rad / 2.0)
    f_y = f_x
    c_x = width / 2.0
    c_y = height / 2.0
    return np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])


def camera_to_world(pos_cam: np.ndarray, view_matrix) -> np.ndarray:
    """
    相机坐标 → 世界坐标（单点）。
    pos_cam = [x, y, -z_real]，OpenGL 约定：Z 朝屏幕外（相机背后）。
    """
    V = np.array(view_matrix, dtype=np.float64).reshape(4, 4).T
    V_inv = np.linalg.inv(V)
    pos_h = np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0])
    world_h = V_inv @ pos_h
    return world_h[:3] / world_h[3]


def _pixels_to_world_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    view_matrix,
) -> np.ndarray | None:
    """
    一批像素 + 深度 → 3D 世界坐标点 (N, 3)。
    """
    n = len(zs)
    if n == 0:
        return None
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = -zs
    V = np.array(view_matrix, dtype=np.float64).reshape(4, 4).T
    V_inv = np.linalg.inv(V)
    cam_h = np.column_stack([x_cam, y_cam, z_cam, np.ones(n, dtype=np.float64)])
    world_h = (V_inv @ cam_h.T).T
    return world_h[:, :3] / world_h[:, 3:4]


def _pixels_to_world_centroid(
    xs, ys, zs, fx, fy, cx, cy, view_matrix,
) -> np.ndarray | None:
    """像素点云 → 世界坐标质心（单视角）。"""
    pts = _pixels_to_world_points(xs, ys, zs, fx, fy, cx, cy, view_matrix)
    if pts is None:
        return None
    return np.mean(pts, axis=0)


class PoseEstimator:
    """
    mask + depth → 点云质心 位姿估计器。
    支持单视角 (estimate) 和多视角融合 (estimate_multiview)。
    """

    def __init__(self, cy_offset: int = -25, verbose: bool = True):
        fov_rad = np.deg2rad(CAM_FOV_DEG)
        self.fx = (CAM_WIDTH / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = self.fx
        self.cx = CAM_WIDTH / 2.0
        self.cy = CAM_HEIGHT / 2.0 + cy_offset
        self.cy_offset = cy_offset

        if verbose:
            print(f"✅ [PoseEstimator] 点云质心模式, cy_offset={cy_offset}")

    def extract_view_points(
        self, depth, seg_img, view_matrix, obj_id
    ) -> np.ndarray | None:
        """
        从单个视角提取物体像素并反投影为世界点云 (N, 3)。

        NOTE: obj_id 来自 PyBullet seg_img，是仿真特有的 GT 分割。
        真实场景中应将此方法替换为 YOLO mask 输入（参见 detector.detect_all）。
        """
        obj_mask = seg_img == obj_id
        if not np.any(obj_mask):
            return None
        ys, xs = np.where(obj_mask)
        z_all = depth[ys, xs]
        valid = (
            (z_all > CAM_NEAR + 0.005)
            & (z_all < CAM_FAR - 0.005)
            & np.isfinite(z_all)
        )
        if valid.sum() < 10:
            return None
        return _pixels_to_world_points(
            xs[valid].astype(np.float64),
            ys[valid].astype(np.float64),
            z_all[valid].astype(np.float64),
            self.fx, self.fy, self.cx, self.cy,
            view_matrix,
        )

    def get_point_cloud_stats(
        self, depth, seg_img, view_matrix, obj_id
    ) -> dict | None:
        """
        点云统计分析，用于优化抓取参数。

        Returns
        -------
        dict: centroid, z_min, z_max, z_mid, z_range,
              xy_extent (水平跨度), long_axis (主方向)
        """
        pts = self.extract_view_points(depth, seg_img, view_matrix, obj_id)
        if pts is None:
            return None
        centroid = np.mean(pts, axis=0)
        z_vals = pts[:, 2]
        xy = pts[:, :2] - centroid[:2]
        cov = np.cov(xy.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        long_idx = np.argmax(eigenvalues)
        return {
            "centroid": centroid,
            "z_min": float(z_vals.min()),
            "z_max": float(z_vals.max()),
            "z_mid": float((z_vals.min() + z_vals.max()) / 2),
            "z_range": float(z_vals.max() - z_vals.min()),
            "xy_std": float(np.sqrt(np.sum(eigenvalues))),
            "long_axis": eigenvectors[:, long_idx],
            "num_points": int(len(pts)),
        }

    def estimate(
        self,
        depth: np.ndarray,
        seg_img: np.ndarray,
        view_matrix,
        obj_id: int,
    ) -> dict | None:
        """
        单视角 mask 像素 → 3D 点云 → 世界坐标质心。
        """
        pts = self.extract_view_points(depth, seg_img, view_matrix, obj_id)
        if pts is None:
            return None
        return {
            "position_world": np.mean(pts, axis=0),
            "num_points": int(len(pts)),
        }

    @staticmethod
    def auto_calibrate_cy(
        views: list[tuple],
        obj_id: int,
        offset_range=None,
        table_z: float | None = None,
    ) -> dict:
        """
        自标定 camera cy 偏移（不依赖 GT，Leave-One-Out 交叉验证 + 桌面约束）。

        对每个候选 cy_offset：
        1. 计算所有 N 个视角的质心 + 点云 Z_min
        2. LOO: 每个视角 i 距其余视角共识的距离均值
        3. 桌面约束: 点云底部应接近桌面高度（若提供 table_z）
        4. 综合评分 = LOO误差 + λ * 桌面偏差

        LOO 确保视角间一致性，桌面约束锚定绝对 Z 轴。
        """
        if offset_range is None:
            offset_range = range(-120, 20, 2)

        best_offset = 0
        best_score = float("inf")

        for offset in offset_range:
            est = PoseEstimator(cy_offset=offset, verbose=False)
            centroids = []
            z_mins = []
            for depth, seg_img, view_matrix in views:
                pts = est.extract_view_points(depth, seg_img, view_matrix, obj_id)
                if pts is not None and len(pts) >= 30:
                    centroids.append(np.mean(pts, axis=0))
                    z_mins.append(float(pts[:, 2].min()))

            if len(centroids) < 3:
                continue

            centroids = np.array(centroids)
            # LOO on XY only — azimuthally diverse views give meaningful XY consensus
            loo_errors = []
            for i in range(len(centroids)):
                others = np.delete(centroids, i, axis=0)
                consensus = np.median(others, axis=0)
                loo_errors.append(float(np.linalg.norm(centroids[i][:2] - consensus[:2])))

            score = float(np.mean(loo_errors))

            # Z anchored by table plane — LOO can't detect common Z bias
            if table_z is not None and len(z_mins) >= 3:
                median_z_min = float(np.median(z_mins))
                table_penalty = abs(median_z_min - table_z - 0.02)
                score += 8.0 * table_penalty

            if score < best_score:
                best_score = score
                best_offset = offset

        return {
            "cy_offset": best_offset,
            "loo_error_m": best_score,
        }

    def estimate_multiview(
        self,
        views: list[tuple],
        obj_id: int,
        quality_weighted: bool = True,
    ) -> dict | None:
        """
        多视角点云融合（逐视角质心 → LOO 质量加权中位数）。

        1. 每个视角独立计算 3D 质心
        2. Leave-One-Out 评估各视角质量：距"其余视角中位数"越近=质量越高
        3. 低质量视角降权，加权中位数融合

        Parameters
        ----------
        views : list of (depth, seg_img, view_matrix)
        obj_id : 物体 ID
        quality_weighted : 启用 LOO 质量加权

        Returns
        -------
        dict: position_world, total_points, views_used, per_view_centroids,
              per_view_quality
        """
        per_view = []
        total_pts = 0
        for depth, seg_img, view_matrix in views:
            pts = self.extract_view_points(depth, seg_img, view_matrix, obj_id)
            if pts is not None and len(pts) >= 30:
                per_view.append(np.mean(pts, axis=0))
                total_pts += len(pts)

        if not per_view:
            print("⚠️ [PoseEstimator] 所有视角均无有效点云")
            return None

        centroids = np.array(per_view)
        n = len(centroids)

        # LOO 质量评分：每个视角距"其余视角中位数"的距离
        loo_scores = np.ones(n, dtype=np.float64)
        if quality_weighted and n >= 3:
            for i in range(n):
                others = np.delete(centroids, i, axis=0)
                consensus = np.median(others, axis=0)
                loo_scores[i] = 1.0 / max(
                    float(np.linalg.norm(centroids[i] - consensus)), 0.001
                )
            loo_scores /= loo_scores.sum()

        if quality_weighted and n >= 3:
            # 加权中位数：每维度独立，按权重排序累积过半处为 median
            position = np.zeros(3)
            for d in range(3):
                order = np.argsort(centroids[:, d])
                cum_w = np.cumsum(loo_scores[order])
                idx = min(np.searchsorted(cum_w, 0.5), n - 1)
                position[d] = centroids[order[idx], d]
        else:
            position = np.median(centroids, axis=0)

        return {
            "position_world": position,
            "total_points": int(total_pts),
            "views_used": n,
            "per_view_centroids": centroids,
            "per_view_quality": loo_scores.tolist(),
        }

    def get_multiview_point_cloud_stats(
        self,
        views: list[tuple],
        obj_id: int,
    ) -> dict | None:
        """
        合并多视角点云后做几何分析。

        比单视角点云更完整（各视角看到物体不同面），
        返回的 Z 范围和 PCA 主轴更准确。
        """
        all_pts = []
        for depth, seg_img, view_matrix in views:
            pts = self.extract_view_points(depth, seg_img, view_matrix, obj_id)
            if pts is not None and len(pts) >= 30:
                all_pts.append(pts)

        if not all_pts:
            return None

        merged = np.vstack(all_pts)
        centroid = np.mean(merged, axis=0)
        z_vals = merged[:, 2]
        xy = merged[:, :2] - centroid[:2]
        cov = np.cov(xy.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        long_idx = np.argmax(eigenvalues)

        return {
            "centroid": centroid,
            "z_min": float(z_vals.min()),
            "z_max": float(z_vals.max()),
            "z_mid": float((z_vals.min() + z_vals.max()) / 2),
            "z_range": float(z_vals.max() - z_vals.min()),
            "xy_std": float(np.sqrt(np.sum(eigenvalues))),
            "long_axis": eigenvectors[:, long_idx],
            "num_points": int(len(merged)),
        }

