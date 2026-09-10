# scripts/test_pose_estimation.py
# 独立测试：位姿估计方案2（centroid）完整流程验证

import math
import os
import sys

import cv2
import numpy as np
import pybullet as p
import pybullet_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import (
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_WIDTH,
    OBJ_INIT_POS,
    TABLE_POS,
)
from src.perception.camera import get_camera_image
from src.perception.pose_estimator import PoseEstimator, get_camera_intrinsics


def build_mask_from_seg(seg_img: np.ndarray, obj_id: int) -> np.ndarray:
    """从 PyBullet 分割图生成二值 mask"""
    return (seg_img == obj_id).astype(np.uint8)


def visualize_pose_result(rgb_bgr, result, save_path="assets/pose_result.png"):
    """在图像上叠加位姿估计结果可视化"""
    vis = rgb_bgr.copy()
    if result is None:
        cv2.imwrite(save_path, vis)
        return

    # 叠加文字信息
    p_w = result["position_world"]
    lines = [
        f"World: ({p_w[0]:.3f}, {p_w[1]:.3f}, {p_w[2]:.3f})",
    ]
    if result.get("error_m") is not None:
        lines.append(f"Error: {result['error_m'] * 100:.1f} cm")
    if result.get("num_points"):
        lines.append(f"Points: {result['num_points']}")

    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (10, 25 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 100),
            2,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)
    print(f"✅ 可视化已保存: {save_path}")


def main():
    print("🚀 启动位姿估计测试环境...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=TABLE_POS)

    # 加载鸭子，修正朝向
    duck_quat = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    obj_id = p.loadURDF(
        "duck_vhacd.urdf",
        basePosition=OBJ_INIT_POS,
        baseOrientation=duck_quat,
        globalScaling=1.5,
    )

    # 等待物理稳定
    for _ in range(240):
        p.stepSimulation()

    print("📸 采集相机数据...")
    rgb, depth_real, depth_vis, seg, view_matrix, proj_matrix = get_camera_image()

    # 保存原始图像备查
    os.makedirs("assets", exist_ok=True)
    cv2.imwrite("assets/pose_rgb.png", rgb)
    cv2.imwrite("assets/pose_depth.png", depth_vis)

    # 从分割图生成 mask
    mask = build_mask_from_seg(seg, obj_id)
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite("assets/pose_mask.png", mask_vis)
    print(f"   分割 mask 有效像素数: {mask.sum()}")

    # ── 方案：点云质心估计 ───────────────────────────────────────────
    print("\n─── 点云质心估计 ───")
    estimator = PoseEstimator()
    result = estimator.estimate(
        depth=depth_real,
        seg_img=seg,
        view_matrix=view_matrix,
        obj_id=obj_id,
    )

    if result is not None:
        print(f"   估计位置: X={result['position_world'][0]:.4f}  "
              f"Y={result['position_world'][1]:.4f}  "
              f"Z={result['position_world'][2]:.4f}")
        print(f"   有效点数: {result['num_points']}")
        visualize_pose_result(rgb, result, "assets/pose_result_centroid.png")
        print("   ✅ 位姿估计成功")
    else:
        print("   ⚠️  位姿估计失败")

    # ── 打印内参矩阵供后续参考 ────────────────────────────────────────
    K = get_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV_DEG)
    print(f"\n📷 相机内参矩阵 K:\n{K}")

    print("\n🎉 位姿估计测试完成！检查 assets/ 目录查看结果图像。")
    print("   - assets/pose_rgb.png        RGB 原图")
    print("   - assets/pose_depth.png      深度可视化")
    print("   - assets/pose_mask.png       分割 mask")
    print("   - assets/pose_result_centroid.png  质心估计结果标注")

    input("\n按 Enter 关闭仿真窗口...")
    p.disconnect()


if __name__ == "__main__":
    main()
