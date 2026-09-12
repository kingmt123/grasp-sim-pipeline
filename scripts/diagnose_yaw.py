# scripts/diagnose_yaw.py
"""
6D 姿态（偏航）可行性诊断 —— 决策卡的证据来源（只读，不改主线）。

背景
----
主线偏航是"物体类别硬编码"（duck 0° / cube 0° / teddy 90°），理由写在 STEP7：
"PCA 在单视角（透视畸变制造虚假长轴）和多视角（点云圆形化）均不可行"。
但该结论是在 STEP12 修正内参之前得出的 —— 当时点云 Z 偏高 8~12cm、y 轴镜像、
逐视角 X/Y 展布 2~3cm，PCA 当然不稳。

本脚本要回答三个问题（用 GT 对比，不参与估计）：
  T1 融合点云的形状还能不能区分长轴？  → XY 展布、伸长比 λ1/λ2、minAreaRect 长宽比
  T2 PCA 角度能否跟随物体真实旋转？    → 把物体绕 Z 逐档旋转 0/30/60/90/135°，
                                        看 PCA 角是否同步旋转（追踪误差）
  T3 现在硬编码偏航错多少？            → 旋转后硬编码值 vs 几何长轴/GT 姿态的夹角

用法:
    uv run python scripts/diagnose_yaw.py                 # YOLO mask（贴近主线）
    uv run python scripts/diagnose_yaw.py --gt-mask        # PyBullet seg mask（上界参考）
    uv run python scripts/diagnose_yaw.py --jitter 0.02 --trials 3
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import pybullet as p  # noqa: E402

from src.configs.config import (  # noqa: E402
    CAM_FAR,
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_NEAR,
    CAM_WIDTH,
    MULTI_OBJECTS_CONFIG,
    TABLE_SURFACE_Z,
)
from src.env.sim_env import init_panda_pose, setup_simulation_multi, stabilize_objects  # noqa: E402
from src.perception.camera import get_camera_image  # noqa: E402
from src.perception.pose_estimator import _pixels_to_world_points  # noqa: E402

CLASS_IDS = {"duck": 0, "teddy": 1, "cube": 2}

FUSION_EYES = [[1.3, 0.0, 1.4], [0.5, -1.0, 1.2], [-0.2, -0.5, 1.0], [0.8, 0.8, 1.5]]
APPLIED_YAWS_DEG = [0.0, 30.0, 60.0, 90.0, 135.0]
# 主线当前硬编码偏航（rad）
HARDCODED_YAW = {"duck": 0.0, "teddy": math.pi / 2, "cube": 0.0}


def wrap180(deg: float) -> float:
    return (deg + 90.0) % 180.0 - 90.0


def cloud_yaw_deg(pts_xy: np.ndarray) -> tuple:
    """点云 XY 投影 → (PCA 长轴角 deg, 伸长比, minAreaRect 角 deg, 长宽比)。"""
    c = pts_xy - pts_xy.mean(axis=0)
    cov = np.cov(c.T)
    evals, evecs = np.linalg.eigh(cov)          # 升序
    major = evecs[:, -1]
    pca_ang = math.degrees(math.atan2(major[1], major[0]))
    elong = float(np.sqrt(evals[-1] / max(evals[0], 1e-12)))
    rect = (cv2.minAreaRect(pts_xy.astype(np.float32)))
    (_, _), (rw, rh), rang = rect
    long_side_ang = rang if rw >= rh else rang + 90.0
    aspect = float(max(rw, rh) / max(min(rw, rh), 1e-9))
    return pca_ang, elong, long_side_ang, aspect


def gt_principal_axes(obj_id: int) -> tuple:
    """GT: 基座朝向(世界) ∘ 惯量局部姿态 → 主惯量轴在世界系的方向。"""
    pos, quat = p.getBasePositionAndOrientation(obj_id)
    euler = p.getEulerFromQuaternion(quat)
    dyn = p.getDynamicsInfo(obj_id, -1)
    inertia_diag = dyn[2]
    local_orn = dyn[4]
    Rb = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    Rl = np.array(p.getMatrixFromQuaternion(local_orn)).reshape(3, 3)
    R = Rb @ Rl
    axes = [R[:, i] for i in range(3)]
    order = np.argsort(list(inertia_diag))       # 惯量小 = 伸长方向
    return {
        "base_yaw_deg": math.degrees(euler[2]),
        "inertia_diag": [round(float(v), 8) for v in inertia_diag],
        "longest_axis_xy_deg": math.degrees(math.atan2(axes[order[0]][1], axes[order[0]][0])),
        "shortest_axis_xy_deg": math.degrees(math.atan2(axes[order[2]][1], axes[order[2]][0])),
    }


def build_cloud(obj_name, class_id, obj_id, detector, gt_mask, applied_xy):
    """4 视角（相机指向物体）→ 点云合并（世界系）。返回 (pts, 各视角点数, 有效视角数)。"""
    per_view, counts = [], []
    tgt = [applied_xy[0], applied_xy[1], TABLE_SURFACE_Z + 0.02]
    for eye in FUSION_EYES:
        rgb, depth, _, seg, vm, _ = get_camera_image(eye=eye, target=tgt)
        if gt_mask:
            mask = (seg == obj_id)
        else:
            dets = detector.detect_all(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            t = next((d for d in dets if d["cls_id"] == class_id), None)
            if t is None:
                per_view.append(None)
                counts.append(0)
                continue
            m = cv2.resize(t["mask"], (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_NEAREST)
            mask = (m > 0.5)
        mask = mask & (depth > CAM_NEAR + 0.005) & (depth < CAM_FAR - 0.005)
        if mask.sum() < 30:
            per_view.append(None)
            counts.append(int(mask.sum()))
            continue
        ys, xs = np.where(mask)
        pts = _pixels_to_world_points(
            xs.astype(np.float64), ys.astype(np.float64),
            depth[ys, xs].astype(np.float64),
            (CAM_HEIGHT / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0),
            (CAM_HEIGHT / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0),
            CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0, vm,
        )
        per_view.append(pts)
        counts.append(int(mask.sum()))
    good = [v for v in per_view if v is not None]
    cloud = np.vstack(good) if good else None
    return cloud, counts, len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="duck,teddy,cube")
    ap.add_argument("--yaws", default="0,30,60,90,135")
    ap.add_argument("--gt-mask", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    from src.perception.detector import YoloSegmentor

    objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    yaws = [float(s) for s in args.yaws.split(",") if s.strip()]
    model_path = os.path.join(REPO_ROOT, "models", "custom_yolov8n_seg.pt")
    det = None if args.gt_mask else YoloSegmentor(model_path=model_path)

    print("=" * 96)
    print("  🧭 6D 姿态（偏航）可行性诊断")
    print(f"  物体={objects}  施加偏航={yaws}°  mask源={'GT seg' if args.gt_mask else 'YOLO'}"
          f"  抖动={args.jitter}m  trials={args.trials}")
    print("=" * 96)

    rng = np.random.default_rng(42)
    rows = []
    for t in range(args.trials):
        for yaw_deg in yaws:
            cfgs = []
            for c in MULTI_OBJECTS_CONFIG:
                cfg = dict(c)
                jx = float(rng.uniform(-args.jitter, args.jitter)) if args.jitter else 0.0
                jy = float(rng.uniform(-args.jitter, args.jitter)) if args.jitter else 0.0
                cfg["pos"] = [c["pos"][0] + jx, c["pos"][1] + jy, c["pos"][2]]
                cfg["euler"] = [c["euler"][0], c["euler"][1], c["euler"][2] + yaw_deg]
                cfgs.append(cfg)
            p.connect(p.DIRECT)
            _, robot_id, obj_infos = setup_simulation_multi(gui=False, objects_config=cfgs)
            init_panda_pose(robot_id)
            # stabilize_objects 会把朝向重置为单位四元数 → 之后重新施加配置朝向
            stabilize_objects(obj_infos, steps=240)
            for info, cfg in zip(obj_infos, cfgs):
                quat = p.getQuaternionFromEuler([math.radians(a) for a in cfg["euler"]])
                pos_now = p.getBasePositionAndOrientation(info["obj_id"])[0]
                p.resetBasePositionAndOrientation(info["obj_id"], pos_now, quat)
                p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])
            for _ in range(120):
                p.stepSimulation()

            for info, cfg in zip(obj_infos, cfgs):
                name = info["name"]
                if name not in objects:
                    continue
                gtp = gt_principal_axes(info["obj_id"])
                cloud, counts, nview = build_cloud(
                    name, CLASS_IDS[name], info["obj_id"], det, args.gt_mask,
                    cfg["pos"][:2])
                if cloud is None or len(cloud) < 100:
                    print(f"  [{name}] 点云不足（{nview} 视角）")
                    continue
                pca_ang, elong, rect_ang, aspect = cloud_yaw_deg(cloud[:, :2])
                # PCA 角是"长轴"，抓手要垂直于长轴（夹窄处）→ 抓手偏航 = 长轴 + 90°
                grip_from_cloud = pca_ang + 90.0
                grip_hard = math.degrees(HARDCODED_YAW.get(name, 0.0))
                d_hard_cloud = wrap180(grip_hard - grip_from_cloud)
                d_hard_gt = wrap180(grip_hard - (gtp["shortest_axis_xy_deg"] + 90.0))
                d_cloud_gt = wrap180(grip_from_cloud - (gtp["shortest_axis_xy_deg"] + 90.0))
                print(f"  [{name:<6}] 施加{yaw_deg:6.1f}° | 点云 n={len(cloud):>6} 视角={nview} "
                      f"伸长比={elong:4.2f} minRect长宽比={aspect:4.2f} | "
                      f"PCA长轴={pca_ang:+7.1f}° → 建议偏航={grip_from_cloud:+7.1f}° | "
                      f"GT基座偏航={gtp['base_yaw_deg']:+7.1f}° GT短轴偏航="
                      f"{wrap180(gtp['shortest_axis_xy_deg'] + 90):+7.1f}° | "
                      f"硬编码{grip_hard:+6.1f}° 误差: 对点云{d_hard_cloud:+6.1f}° "
                      f"对GT{d_hard_gt:+6.1f}° | 点云vs GT误差={d_cloud_gt:+6.1f}°")
                rows.append({"trial": t, "object": name, "applied_yaw_deg": yaw_deg,
                             "n_pts": int(len(cloud)), "n_views": nview,
                             "elongation": round(elong, 3), "rect_aspect": round(aspect, 3),
                             "pca_long_axis_deg": round(pca_ang, 2),
                             "grip_yaw_from_cloud_deg": round(grip_from_cloud, 2),
                             "gt_base_yaw_deg": round(gtp["base_yaw_deg"], 2),
                             "gt_grip_yaw_deg": round(wrap180(gtp["shortest_axis_xy_deg"] + 90), 2),
                             "hardcoded_grip_deg": round(grip_hard, 2),
                             "err_hard_vs_cloud_deg": round(d_hard_cloud, 2),
                             "err_hard_vs_gt_deg": round(d_hard_gt, 2),
                             "err_cloud_vs_gt_deg": round(d_cloud_gt, 2)})
            p.disconnect()

    print("\n" + "=" * 96)
    print("  汇总（按物体）：追踪误差 = 点云建议偏航 vs GT 建议偏航 的绝对值中位数")
    print("=" * 96)
    for name in objects:
        rs = [r for r in rows if r["object"] == name]
        if not rs:
            continue
        ec = np.median([abs(r["err_cloud_vs_gt_deg"]) for r in rs])
        eh = np.median([abs(r["err_hard_vs_gt_deg"]) for r in rs])
        el = np.median([r["elongation"] for r in rs])
        ar = np.median([r["rect_aspect"] for r in rs])
        print(f"  {name:<6} 伸长比中位={el:4.2f} minRect长宽比={ar:4.2f} | "
              f"点云偏航误差中位={ec:5.1f}°  硬编码偏航误差中位={eh:5.1f}°  "
              f"→ {'点云可用' if ec < 10 else '点云仍不可用'}")

    runs = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    path = os.path.join(runs, f"yaw_diag_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"mask_source": "gt_seg" if args.gt_mask else "yolo",
                   "jitter_m": args.jitter, "trials": args.trials,
                   "applied_yaws_deg": yaws, "rows": rows,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n💾 明细 → {path}")


if __name__ == "__main__":
    main()
