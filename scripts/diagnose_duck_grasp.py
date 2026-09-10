# scripts/diagnose_duck_grasp.py
"""
诊断鸭子抓取问题：从生成位姿到物体角度逐层排查。
"""

import math
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import MULTI_OBJECTS_CONFIG, TABLE_SURFACE_Z
from src.env.sim_env import setup_simulation_multi, stabilize_objects
from src.perception.camera import get_camera_image
from src.perception.pose_estimator import PoseEstimator

TABLE_Z = TABLE_SURFACE_Z
FUSION_VIEWS = [
    {"eye": [1.3,  0.0, 1.4]},
    {"eye": [0.5, -1.0, 1.2]},
    {"eye": [-0.2,-0.5, 1.0]},
]
FIXED_CY = -6


def main():
    print("=" * 60)
    print("  🦆 鸭子抓取问题诊断")
    print("=" * 60)

    # ── 1. 物理特性 ─────────────────────────────────────────────
    print("\n--- 1. 物理特性 ---")
    client, robot_id, obj_infos = setup_simulation_multi(gui=False)

    # 加载后初始位姿
    print("\n加载后初始位姿:")
    for info in obj_infos:
        pos, quat = p.getBasePositionAndOrientation(info["obj_id"])
        euler = p.getEulerFromQuaternion(quat)
        print(f"  [{info['name']}] ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
              f"euler_deg=({math.degrees(euler[0]):.0f},{math.degrees(euler[1]):.0f},{math.degrees(euler[2]):.0f})")

    # 稳定后
    stabilize_objects(obj_infos, steps=240)
    print("\n稳定后位姿:")
    for info in obj_infos:
        pos, quat = p.getBasePositionAndOrientation(info["obj_id"])
        euler = p.getEulerFromQuaternion(quat)
        dyn = p.getDynamicsInfo(info["obj_id"], -1)
        print(f"  [{info['name']}] ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
              f"mass={dyn[0]:.1f}kg  friction={dyn[1]:.2f}  restitution={dyn[5]:.2f}")

    # 默认视角分割像素
    _, _, _, seg, _, _ = get_camera_image()
    print("\n默认视角分割像素:")
    for info in obj_infos:
        px = np.sum(seg == info["obj_id"])
        print(f"  [{info['name']}] {px} px")
    p.disconnect()

    # ── 2. cy_offset 灵敏度 ────────────────────────────────────
    print("\n--- 2. cy_offset 对不同物体的 Z 估计影响 ---")

    def test_cy_offsets():
        client, robot_id, obj_infos = setup_simulation_multi(gui=False)
        stabilize_objects(obj_infos, steps=120)
        _, depth, _, seg, vm, _ = get_camera_image()

        for info in obj_infos:
            name, oid = info["name"], info["obj_id"]
            gt = p.getBasePositionAndOrientation(oid)[0]
            print(f"\n  [{name}] GT Z={gt[2]:.3f}")
            print(f"  {'cy_offset':>10s}  {'est_Z':>8s}  {'Z_err':>8s}  {'total':>8s}")
            for offset in range(-100, 20, 20):
                est = PoseEstimator(cy_offset=offset, verbose=False)
                r = est.estimate(depth, seg, vm, oid)
                if r is None:
                    continue
                stats = est.get_point_cloud_stats(depth, seg, vm, oid)
                if stats:
                    r["position_world"][2] = stats["z_mid"]
                z_err = (r["position_world"][2] - gt[2]) * 100
                total = np.linalg.norm(r["position_world"] - np.array(gt)) * 100
                print(f"  {offset:+8d}   {r['position_world'][2]:.3f}  {z_err:+7.1f}cm  {total:>6.1f}cm")
        p.disconnect()

    test_cy_offsets()

    # ── 3. 多视角最佳 cy_offset ─────────────────────────────────
    print("\n--- 3. 各物体在融合视角下的最佳 cy_offset ---")

    def best_cy_for_object(oid, rough, info_name):
        views = []
        for vc in FUSION_VIEWS:
            _, d, _, s, vm, _ = get_camera_image(eye=vc["eye"], target=rough)
            views.append((d, s, vm))

        best_cy, best_err = 0, float("inf")
        for offset in range(-120, 20, 2):
            est = PoseEstimator(cy_offset=offset, verbose=False)
            r = est.estimate_multiview(views, oid)
            if r is None:
                continue
            stats = est.get_multiview_point_cloud_stats(views, oid)
            if stats:
                r["position_world"][2] = stats["z_mid"]
            gt = np.array(p.getBasePositionAndOrientation(oid)[0])
            err = np.linalg.norm(r["position_world"] - gt) * 100
            if err < best_err:
                best_err = err
                best_cy = offset
        return best_cy, best_err

    client, robot_id, obj_infos = setup_simulation_multi(gui=False)
    stabilize_objects(obj_infos, steps=120)
    for info in obj_infos:
        rough = list(p.getBasePositionAndOrientation(info["obj_id"])[0])
        cy, err = best_cy_for_object(info["obj_id"], rough, info["name"])
        print(f"  [{info['name']}] 最佳 cy={cy:+d}  融合误差={err:.1f}cm")

    # 固定 cy=-6 的效果
    print(f"\n  固定 cy={FIXED_CY:+d} 的结果:")
    for info in obj_infos:
        oid, name = info["obj_id"], info["name"]
        rough = list(p.getBasePositionAndOrientation(oid)[0])
        views = []
        for vc in FUSION_VIEWS:
            _, d, _, s, vm, _ = get_camera_image(eye=vc["eye"], target=rough)
            views.append((d, s, vm))
        est = PoseEstimator(cy_offset=FIXED_CY)
        r = est.estimate_multiview(views, oid)
        if r:
            stats = est.get_multiview_point_cloud_stats(views, oid)
            if stats:
                r["position_world"][2] = stats["z_mid"]
            gt = np.array(p.getBasePositionAndOrientation(oid)[0])
            err = np.linalg.norm(r["position_world"] - gt) * 100
            z_err = (r["position_world"][2] - gt[2]) * 100
            print(f"  [{name}] err={err:.1f}cm  Z_err={z_err:+.1f}cm  "
                  f"est_Z={r['position_world'][2]:.3f}  GT_Z={gt[2]:.3f}")
    p.disconnect()

    # ── 4. 总结 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🔍 诊断结论")
    print("=" * 60)
    print("""
   duck 抓取失败根因分析（scaling=1.0）：

   1. 尺寸问题
      - duck 稳定后 Z=~0.642，高度仅 ~7.3cm
      - cube 高度 ~6.6cm（一样小但 cube 是平面好抓）
      - teddy 高度 ~8.9cm（最大，35% 抓取点宽容度高）

   2. cy_offset 敏感度
      - duck 最佳 cy_offset 约 -14~-30（与 cube 的 -6 不同）
      - 用固定 cy=-6 时 duck 的 Z 偏差约 +4cm（估高了）
      - teddy 对 cy_offset 不敏感（体型大）
      - cube 对 cy_offset 最敏感但形状规则好补偿

   3. 抓取几何
      - duck 1.0x 只有 7.3cm 高，IK 误差 3.9cm = 53% 的物体高度
      - 实际指尖位置比目标高 ~4cm(Z偏差) + ~4cm(IK误差) - 0.5cm(修正)
        ≈ 7.5cm 偏高 → 落在鸭子脖子/头部区域（窄）
      - teddy 9cm 高 + 35% 点 + 800N = 即使 Z 偏差也能抓
      - cube 平面底部 + 精准 cy_offset = 即使 Z 偏差也能 hold

   4. 改进方案
      A. 为 duck 单独标定 cy_offset：使用 duck 自身的 auto_calibrate
      B. duck grasp 比率从 25% 改到 15%（更低，更靠近宽腹部）
      C. 先算出最佳 cy 再用多视角融合（不复标定）
   """)


if __name__ == "__main__":
    main()
