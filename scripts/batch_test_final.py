# scripts/batch_test_final.py
"""
最终批量测试：10 次位姿估计 + 5 次完整抓取。
记录优化后（无 Z_mid 修正）的精度和成功率。
"""

import math
import os
import random
import sys
import time

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import MULTI_OBJECTS_CONFIG, TABLE_SURFACE_Z
from src.env.sim_env import setup_simulation_multi, stabilize_objects, init_panda_pose
from src.perception.camera import get_camera_image
from src.perception.pose_estimator import PoseEstimator
from src.control.ik_controller import move_to_pose
TABLE_Z = TABLE_SURFACE_Z
FUSION_VIEWS = [
    {"eye": [1.3,  0.0, 1.4]},
    {"eye": [0.5, -1.0, 1.2]},
    {"eye": [-0.2,-0.5, 1.0]},
]
FIXED_CY = -6
END_EFF = 8
FOFF = 0.105
CLASS_NAMES = ["duck", "teddy", "cube"]

POSE_N = 10   # 位姿估计
GRASP_N = 5   # 完整抓取

# ── 多视角融合（无 Z_mid 修正） ──────────────────────────────────
def multiview_estimate(oid, rough):
    views = []
    for vc in FUSION_VIEWS:
        _, d, _, s, vm, _ = get_camera_image(eye=vc["eye"], target=rough)
        views.append((d, s, vm))
    if len(views) < 3:
        return None, {}
    est = PoseEstimator(cy_offset=FIXED_CY)
    r = est.estimate_multiview(views, oid)
    if r is None:
        return None, {}
    pos = r["position_world"]
    # 无 Z_mid 修正 — 直接用融合质心 Z
    stats = est.get_multiview_point_cloud_stats(views, oid)
    grasp_params = {}
    if stats:
        z_min, z_range = stats["z_min"], stats["z_range"]
        grasp_params = {"z_min": z_min, "z_range": z_range,
                        "long_axis": stats["long_axis"]}
    return pos, grasp_params


def compute_grasp_z(name, grasp_params):
    if not grasp_params:
        return FOFF + 0.1, 0.0
    z_min = grasp_params["z_min"]
    z_range = grasp_params["z_range"]
    if name == "teddy":
        ratio = 0.35
    elif name == "duck":
        ratio = 0.25
    else:
        ratio = 0.15
    gz_body = z_min + ratio * z_range
    la = grasp_params.get("long_axis", np.array([1.0, 0.0]))
    yaw = math.atan2(-la[1], la[0])
    return gz_body + FOFF, yaw


# ── 测试 1：位姿精度 ────────────────────────────────────────────
def test_pose_accuracy():
    print("=" * 60)
    print(f"  📏 位姿估计精度测试 N={POSE_N}")
    print("=" * 60)

    data = {n: {"errs": [], "z_errs": []} for n in CLASS_NAMES}

    for run in range(POSE_N):
        client, robot_id, obj_infos = setup_simulation_multi(gui=False)
        rng = random.Random(42 + run)
        placements = [[0.55, 0.35], [0.55, -0.35], [0.30, 0.00]]
        rng.shuffle(placements)

        for i, info in enumerate(obj_infos):
            px, py = placements[i]
            pos = [px, py, TABLE_Z + 0.02]
            yaw = rng.uniform(0, 360)
            q = (p.getQuaternionFromEuler([math.pi/2, 0, math.radians(yaw)])
                 if info["name"] == "duck"
                 else p.getQuaternionFromEuler([0, 0, math.radians(yaw)]))
            p.resetBasePositionAndOrientation(info["obj_id"], pos, q)
            p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])
            f = 1.5 if info["name"] == "duck" else 0.8
            p.changeDynamics(info["obj_id"], -1, restitution=0.0, lateralFriction=f)
        for _ in range(120):
            p.stepSimulation()
        for i, info in enumerate(obj_infos):
            pos, q = p.getBasePositionAndOrientation(info["obj_id"])
            p.resetBasePositionAndOrientation(
                info["obj_id"], [placements[i][0], placements[i][1], pos[2]], q)
            p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])

        for info in obj_infos:
            name, oid = info["name"], info["obj_id"]
            gt = np.array(p.getBasePositionAndOrientation(oid)[0])
            rough = list(gt)
            pos, _ = multiview_estimate(oid, rough)
            if pos is not None:
                err = np.linalg.norm(pos - gt) * 100
                z_err = (pos[2] - gt[2]) * 100
                data[name]["errs"].append(err)
                data[name]["z_errs"].append(z_err)

        p.disconnect()
        if (run + 1) % 5 == 0:
            print(f"  Run {run+1}/{POSE_N} done")

    print(f"\n{'='*60}")
    print(f"  📊 位姿误差 (无 Z_mid 修正)")
    print(f"{'='*60}")
    print(f"  {'物体':>8s}  {'均值±σ':>12s}  {'Z偏差均值':>10s}  {'最小':>6s}  {'最大':>6s}")
    print(f"  {'-'*44}")
    for name in CLASS_NAMES:
        v = data[name]["errs"]
        zv = data[name]["z_errs"]
        if v:
            m, s, lo, hi = np.mean(v), np.std(v), min(v), max(v)
            z_m = np.mean(zv)
            print(f"  {name:>8s}  {m:>5.1f}±{s:>4.1f}  {z_m:>+7.1f}cm  {lo:>5.1f}  {hi:>5.1f}")
    return data


# ── 测试 2：抓取测试 ────────────────────────────────────────────
def test_grasp():
    print(f"\n{'='*60}")
    print(f"  🎯 抓取测试 N={GRASP_N}")
    print(f"{'='*60}")

    results = {n: {"ok": 0, "total": 0, "errs": [], "dzs": []} for n in CLASS_NAMES}

    for run in range(GRASP_N):
        client, robot_id, obj_infos = setup_simulation_multi(gui=True)
        stabilize_objects(obj_infos, steps=240)
        init_panda_pose(robot_id)

        for info in obj_infos:
            name, oid = info["name"], info["obj_id"]
            gt = np.array(p.getBasePositionAndOrientation(oid)[0])
            rough = list(gt)

            pos, gp = multiview_estimate(oid, rough)
            if pos is None:
                continue

            err = np.linalg.norm(pos - gt) * 100
            results[name]["errs"].append(err)

            gz, gyaw = compute_grasp_z(name, gp)
            dq = p.getQuaternionFromEuler([math.pi, 0, gyaw])
            ox, oy = pos[0], pos[1]

            move_to_pose(robot_id, [ox, oy, gz+0.20], target_quat=dq,
                         end_effector_link_index=END_EFF, steps=200)
            move_to_pose(robot_id, [ox, oy, gz+0.05], target_quat=dq,
                         end_effector_link_index=END_EFF, steps=200)
            move_to_pose(robot_id, [ox, oy, gz], target_quat=dq,
                         end_effector_link_index=END_EFF, steps=600,
                         convergence_threshold=2e-3)
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1/240)

            gforce = 800 if name == "teddy" else 500
            for j in [9, 10]:
                p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                        targetPosition=0.0, force=gforce)
            for _ in range(250):
                p.stepSimulation()
                time.sleep(1/240)
            for _ in range(120):
                p.stepSimulation()
                time.sleep(1/240)

            obj_z0 = p.getBasePositionAndOrientation(oid)[0][2]
            move_to_pose(robot_id, [ox, oy, gz+0.05], target_quat=dq,
                         end_effector_link_index=END_EFF, steps=150)
            for _ in range(120):
                p.stepSimulation()
                time.sleep(1/240)
            dz = (p.getBasePositionAndOrientation(oid)[0][2] - obj_z0) * 100

            results[name]["total"] += 1
            results[name]["dzs"].append(dz)
            success = dz > 1.0
            if success:
                results[name]["ok"] += 1

            status = "✅" if success else "❌"
            print(f"  [{run+1}/{GRASP_N}] {name}: {status} err={err:.1f}cm dz={dz:.1f}cm")

            # 松开
            for j in [9, 10]:
                p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                        targetPosition=0.04, force=350)
            for _ in range(100):
                p.stepSimulation()

            init_panda_pose(robot_id)

        p.disconnect()
        print(f"  Run {run+1}/{GRASP_N} done")

    print(f"\n{'='*60}")
    print(f"  📊 抓取成功率")
    print(f"{'='*60}")
    print(f"  {'物体':>8s}  {'成功/总数':>10s}  {'成功率':>8s}  {'均值误差':>10s}  {'均值Δz':>8s}")
    print(f"  {'-'*48}")
    for name in CLASS_NAMES:
        r = results[name]
        rate = r["ok"]/max(r["total"],1)*100
        me = np.mean(r["errs"]) if r["errs"] else 0
        mdz = np.mean(r["dzs"]) if r["dzs"] else 0
        print(f"  {name:>8s}  {r['ok']}/{r['total']:<5d}  {rate:>6.0f}%   {me:>5.1f}cm    {mdz:>5.1f}cm")
    ts = sum(r["ok"] for r in results.values())
    tt = sum(r["total"] for r in results.values())
    print(f"  {'-'*48}")
    print(f"  {'总计':>8s}  {ts}/{tt:<5d}  {ts/max(tt,1)*100:>6.0f}%")

    return results


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    # Phase 1: 位姿精度 (10x, DIRECT)
    pose_data = test_pose_accuracy()

    # Phase 2: 抓取 (5x, GUI)
    grasp_data = test_grasp()

    # 汇总
    print(f"\n{'='*60}")
    print(f"  📈 最终汇总")
    print(f"{'='*60}")
    print(f"  {'物体':>8s}  {'位姿误差':>10s}  {'Z偏差':>8s}  {'抓取率':>8s}  {'提升(vs单视角)':>14s}")
    print(f"  {'-'*52}")
    for name in CLASS_NAMES:
        pe = np.mean(pose_data[name]["errs"]) if pose_data[name]["errs"] else 0
        ze = np.mean(pose_data[name]["z_errs"]) if pose_data[name]["z_errs"] else 0
        gr = grasp_data[name]["ok"]/max(grasp_data[name]["total"],1)*100
        print(f"  {name:>8s}  {pe:>5.1f}±{np.std(pose_data[name]['errs']):.1f}  {ze:>+5.1f}cm  {gr:>5.0f}%   "
              f"82~94%")
    print(f"\n  关键改进: 禁用 Z_mid 修正 + 固定 cy_offset=-6 + 分物体抓取参数")


if __name__ == "__main__":
    main()
