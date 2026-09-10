# scripts/test_multi_object.py
"""
多物体仿真环境测试。
第一步：加载多个物体，验证环境正常。
"""

import os
import sys
import time

import cv2
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import (
    CAM_EYE,
    CAM_FAR,
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_NEAR,
    CAM_TARGET,
    CAM_UP,
    CAM_WIDTH,
)
from src.env.sim_env import setup_simulation_multi, stabilize_objects

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def capture_snapshot(filename="multi_objects_rgb.png"):
    """拍摄当前仿真画面并保存。"""
    import numpy as np
    view_matrix = p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
    proj_matrix = p.computeProjectionMatrixFOV(CAM_FOV_DEG, CAM_WIDTH / CAM_HEIGHT, CAM_NEAR, CAM_FAR)
    img_arr = p.getCameraImage(
        CAM_WIDTH,
        CAM_HEIGHT,
        view_matrix,
        proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    # 使用 np.reshape (兼容 tuple 输入)
    rgb = np.reshape(img_arr[2], (CAM_HEIGHT, CAM_WIDTH, 4)).astype("uint8")[:, :, :3]
    save_path = os.path.join(ASSETS_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"   📸 已保存快照 → {save_path}")
    return rgb


def main():
    print("=" * 60)
    print("  🚀 多物体仿真环境测试 — 第一步")
    print("=" * 60)

    print("\n📦 加载多物体环境...")
    client, robot_id, obj_infos = setup_simulation_multi(gui=True)

    # 多物体稳定
    print("\n🔧 稳定各物体位姿...")
    stabilize_objects(obj_infos, steps=240)

    # 打印最终稳定位置
    print(f"\n📋 物体列表 ({len(obj_infos)} 个):")
    print(f"   {'#':>2s}  {'名称':>8s}  {'obj_id':>6s}  {'初始位置':>30s}  {'实际位置':>30s}")
    print(f"   {'-'*75}")
    for i, info in enumerate(obj_infos):
        pos = info["pos"]
        actual_pos = p.getBasePositionAndOrientation(info["obj_id"])[0]
        print(
            f"   {i:2d}  {info['name']:>8s}  {info['obj_id']:6d}  "
            f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
            f"({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})"
        )

    # 拍摄快照
    print("\n📷 拍摄场景快照...")
    capture_snapshot("multi_objects_rgb.png")

    print("✅ 多物体环境加载成功！")
    print(f"   快照已保存至 assets/multi_objects_rgb.png")

    # 保持仿真 2 秒让用户看 GUI
    for _ in range(480):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    print("👋 自动退出（2秒完毕）")
    p.disconnect()


if __name__ == "__main__":
    main()
