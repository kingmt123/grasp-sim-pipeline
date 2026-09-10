# scripts/diagnose_finger_geometry.py
"""
量取 Panda 末端真实几何：hand 原点、手指 link、指尖 AABB、panda_grasptarget 的实际间距。

目的
----
主线用 FINGER_OFFSET = 0.105 m 把"IK 目标（link 8 panda_hand）"折算成"指尖高度"，
安全钳制 min_fine_z = 桌面 + 0.105 + 5mm 也建立在这个常数上。
但抓取高度扫描显示：按 0.105 折算出的"指尖目标"与实际夹持效果对不上
（cube 在任何 0.60~0.68 的"指尖目标"下都抓不住，而历史上它是"成功"的）。

本脚本直接问 PyBullet 要真实几何，不再假设。

用法: uv run python scripts/diagnose_finger_geometry.py
"""

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pybullet as p  # noqa: E402

from src.control.ik_controller import move_to_pose  # noqa: E402
from src.env.sim_env import init_panda_pose, setup_simulation_multi  # noqa: E402

TARGETS = [[0.50, 0.00, 0.900], [0.50, 0.00, 0.800], [0.50, 0.00, 0.750], [0.55, 0.35, 0.750]]


def main():
    p.connect(p.DIRECT)
    _, robot_id, _ = setup_simulation_multi(gui=False)
    init_panda_pose(robot_id)

    print("=" * 78)
    print("  🔎 Panda 末端真实几何（link 8 = IK 目标 = panda_hand）")
    print("=" * 78)
    n = p.getNumJoints(robot_id)
    print(f"  关节数={n}")
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        print(f"   link {i:2d}  {info[12].decode():<22} joint={info[1].decode():<20}")

    down_quat = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
    for tgt in TARGETS:
        init_panda_pose(robot_id)
        move_to_pose(robot_id, tgt, target_quat=down_quat,
                     end_effector_link_index=8, steps=800, convergence_threshold=1e-3)
        hand = list(p.getLinkState(robot_id, 8)[0])
        st = p.getLinkState(robot_id, 8)
        com = list(st[0])
        frame = list(st[4])          # worldLinkFramePosition = IK 实际控制的 link 坐标系原点
        tip11 = list(p.getLinkState(robot_id, 11)[0]) if n > 11 else [float("nan")] * 3
        fingers = []
        for idx in (9, 10):
            a = p.getAABB(robot_id, linkIndex=idx)
            fingers.append((idx, [round(v, 4) for v in a[0]], [round(v, 4) for v in a[1]]))
        tip_min = min(f[1][2] for f in fingers)
        print(f"\n  ▸ 命令目标 (link8) z={tgt[2]:.3f}")
        print(f"      getLinkState(8)[0] = CoM       z={com[2]:.4f}   (残差 {(tgt[2]-com[2])*100:+.2f}cm)")
        print(f"      getLinkState(8)[4] = link frame z={frame[2]:.4f}   (残差 {(tgt[2]-frame[2])*100:+.2f}cm) ← IK 真正控制的就是它")
        print(f"      link11(panda_grasptarget) z={tip11[2]:.4f}")
        for idx, amin, amax in fingers:
            print(f"      finger link{idx} AABB z=[{amin[2]:.4f}, {amax[2]:.4f}]")
        print(f"      真实指尖最低点 z={tip_min:.4f}  →  真实 FINGER_OFFSET = hand−tip = "
              f"{hand[2]-tip_min:.4f} m   (代码假定 0.1050)")
        print(f"      真实指尖相对桌面(z=0.626) = {(tip_min-0.626)*100:+.2f}cm")
    p.disconnect()


if __name__ == "__main__":
    main()
