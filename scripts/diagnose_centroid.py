# scripts/diagnose_centroid.py
"""
诊断脚本：检查质心反投影的精度问题

运行方式：
  python scripts/diagnose_centroid.py

目的：
  - 对比 GT 位置与反投影位置
  - 可视化分割掩码
  - 分析深度图质量
  - 诊断误差来源
"""

import os
import sys
import time

import cv2
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.env.sim_env import setup_simulation
from src.perception.camera import get_camera_image
from src.perception.pose_estimator import PoseEstimator

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def init_panda_pose(robot_id):
    """初始化 Panda 预备姿态"""
    init_angles = [0, -0.3, 0, -2.0, 0, 1.8, 0.785]
    for i, angle in enumerate(init_angles):
        p.resetJointState(robot_id, i, angle)
    p.resetJointState(robot_id, 9, 0.04)
    p.resetJointState(robot_id, 10, 0.04)
    for _ in range(50):
        p.stepSimulation()


def wait(steps=120):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    print("🔧 启动诊断工具...")
    client, robot_id, obj_id = setup_simulation(gui=True)

    init_panda_pose(robot_id)
    wait(steps=240)

    # 获取 GT 位置
    obj_pos_gt, _ = p.getBasePositionAndOrientation(obj_id)
    print("\n【GT 位置】")
    print(f"  X={obj_pos_gt[0]:.4f}, Y={obj_pos_gt[1]:.4f}, Z={obj_pos_gt[2]:.4f}")

    # 获取相机数据
    print("\n【获取相机数据】")
    rgb_bgr, depth_real, depth_vis, seg_img, view_matrix, proj_matrix = (
        get_camera_image()
    )

    # 保存原始数据
    cv2.imwrite(os.path.join(ASSETS_DIR, "diag_rgb.png"), rgb_bgr)
    cv2.imwrite(os.path.join(ASSETS_DIR, "diag_depth.png"), depth_vis)

    # 【诊断 1】检查分割图
    print("\n【诊断 1：分割图分析】")
    print(f"  分割图形状: {seg_img.shape}")
    print(f"  分割图数据类型: {seg_img.dtype}")
    print(f"  分割图中的唯一 ID: {np.unique(seg_img)}")

    obj_mask = seg_img == obj_id
    valid_pixels = np.sum(obj_mask)
    print(f"  物体 ID={obj_id} 的像素数: {valid_pixels}")

    if valid_pixels == 0:
        print("  ❌ 物体在分割图中未找到！可能的原因：")
        print("     - 物体 ID 不对")
        print("     - 物体不在相机视野内")
        print("     - PyBullet 分割模式有问题")
    else:
        print(f"  ✅ 物体在分割图中，占比 {valid_pixels / (640 * 480) * 100:.1f}%")

        # 保存分割掩码
        mask_vis = (obj_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(ASSETS_DIR, "diag_mask.png"), mask_vis)

    # 【诊断 2】检查深度图
    print("\n【诊断 2：深度图分析】")
    print(f"  深度图范围: {depth_real.min():.3f} - {depth_real.max():.3f} m")
    print(f"  深度图均值: {depth_real.mean():.3f} m")

    if valid_pixels > 0:
        z_obj = depth_real[obj_mask]
        print(f"  物体像素深度范围: {z_obj.min():.3f} - {z_obj.max():.3f} m")
        print(f"  物体像素深度均值: {z_obj.mean():.3f} m")
        print(f"  物体像素深度中位数: {np.median(z_obj):.3f} m")

    # 【诊断 3】执行点云质心反投影
    print("\n【诊断 3：点云质心反投影】")
    estimator = PoseEstimator()

    result = estimator.estimate(
        depth=depth_real,
        seg_img=seg_img,
        view_matrix=view_matrix,
        obj_id=obj_id,
    )

    if result is not None:
        obj_pos = result["position_world"]
        print("\n【反投影结果】")
        print(f"  X={obj_pos[0]:.4f}, Y={obj_pos[1]:.4f}, Z={obj_pos[2]:.4f}")

        # 计算误差
        err_vector = obj_pos - np.array(obj_pos_gt[:3])
        err_dist = np.linalg.norm(err_vector)
        print("\n【误差分析】")
        print(
            f"  误差向量: ΔX={err_vector[0]:.4f}, ΔY={err_vector[1]:.4f}, ΔZ={err_vector[2]:.4f}"
        )
        print(f"  误差距离: {err_dist:.4f} m ({err_dist * 100:.2f} cm)")

        # 【诊断】详细误差分析
        print("\n【诊断：坐标系变换】")
        print(f"   点云质心（世界）: X={obj_pos[0]:.4f}, Y={obj_pos[1]:.4f}, Z={obj_pos[2]:.4f}")
        print(f"   有效点数: {result['num_points']}")

        # 如果 Y 轴误差特别大
        if abs(err_vector[1]) > abs(err_vector[0]) and abs(err_vector[1]) > abs(
            err_vector[2]
        ):
            print(f"\n⚠️  Y 轴误差特别大 ({err_vector[1] * 100:.1f} cm)！")
            print("  可能原因：")
            print("    1. 相机内参 f_y 或 c_y 不准确")
            print("    2. view_matrix 的 Y 轴方向有问题")
            print(f"    3. 相机上向量 CAM_UP={CAM_UP} 设置错误")

        # 【修复建议】
        print("\n【修复建议】")
        if err_dist > 0.1:
            print("  由于误差 > 10 cm，建议尝试以下修复：")
            print("\n  方案 A（快速）：在 pose_estimator.py 中调整内参")
            print("    # 临时调整相机主点")
            print("    self.cy = CAM_HEIGHT / 2.0 - 20  # 尝试向上偏移")

            print("\n  方案 B（彻底）：验证相机参数标定")
            print("    - 确认 CAM_FOV_DEG = 60 是否正确")
            print("    - 确认 CAM_HEIGHT = 480 是否匹配实际图像")
            print("    - 确认 CAM_UP = [0, 0, 1] 是否代表真实的向上方向")

        if result["num_points"] < 500:
            print(f"  有效像素数较少 ({result['num_points']})，可能影响精度")
            print("    - 增大检测框大小")
            print("    - 调整深度范围过滤条件")

        # 生成对比可视化
        vis = rgb_bgr.copy()

        # 画 GT 投影
        # 简单起见，只在图像上标注位置（实际需要投影到像素）
        h, w, _ = vis.shape
        cv2.putText(
            vis,
            f"GT: ({obj_pos_gt[0]:.3f}, {obj_pos_gt[1]:.3f}, {obj_pos_gt[2]:.3f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"EST: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
        cv2.putText(
            vis,
            f"Error: {err_dist * 100:.1f} cm",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if err_dist > 0.1 else (0, 255, 0),
            2,
        )

        # 画估计位置标记
        # 使用图像中心作为参考点标记
        h, w_vis, _ = vis.shape
        cv2.circle(vis, (w_vis // 2, h // 2), 10, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Image Center",
            (w_vis // 2 + 15, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imwrite(os.path.join(ASSETS_DIR, "diag_result.png"), vis)

        print("\n✅ 诊断完成，结果已保存到 assets/ 目录")
        print("   - diag_rgb.png: 原始 RGB 图像")
        print("   - diag_depth.png: 深度图可视化")
        print("   - diag_mask.png: 分割掩码")
        print("   - diag_result.png: 结果对比")

    else:
        print("❌ 质心反投影失败")

    # 自动断开连接，无需等待
    print("\n诊断完成，自动断开连接...")
    p.disconnect()


if __name__ == "__main__":
    main()
