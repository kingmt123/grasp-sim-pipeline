# scripts/diagnose_z_bias.py
"""
Z 系统性偏高根因诊断（只读，不修改主线逻辑）。

已知现象
--------
* 融合质心 Z 恒定比物体 GT 高 +8~12cm（XY 却只有 1.3~1.5cm 误差）。
* 历史上反复用"内参偏差 cy_offset"（-88 → -8 → -54）吸收，换视角/换物体要重标。

本脚本不做任何假设，用 PyBullet 自身作 GT 逐项判定：
  1. 直接从 PyBullet 返回的投影矩阵解析出真实内参 (fx, fy, cx, cy)；
  2. 重投影自一致性：判 f 的倍数；残差在 u/v 上的行为区分"焦距错"与"符号错"；
  3. 物体点云 vs GT AABB：4 组约定 (f 旧/正确 × y 符号 +/−) 里谁与真实 Z 范围吻合；
  4. rayTest：mask 像素的相机射线命中率 —— 约定正确时应接近 100%。

结论（2026-09-10 运行）见文件末尾 SUMMARY 或 runs/z_bias_diag_*.json。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pybullet as p  # noqa: E402

from src.configs.config import (  # noqa: E402
    CAM_EYE,
    CAM_FAR,
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_NEAR,
    CAM_TARGET,
    CAM_WIDTH,
    TABLE_SURFACE_Z,
)
from src.env.sim_env import (  # noqa: E402
    init_panda_pose,
    setup_simulation_multi,
    stabilize_objects,
)
from src.perception.camera import get_camera_image  # noqa: E402

FUSION_EYES = [[1.3, 0.0, 1.4], [0.5, -1.0, 1.2], [-0.2, -0.5, 1.0], [0.8, 0.8, 1.5]]
F_LEGACY = (CAM_WIDTH / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)   # 554.3
F_VERT = (CAM_HEIGHT / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)    # 415.7


def _V(m):
    return np.array(m, dtype=np.float64).reshape(4, 4).T


def unproject(us, vs, zs, K, view_matrix, sign_y=+1.0):
    """像素+深度 → 世界点。sign_y=+1 为主线现有约定，-1 为翻转 y 轴约定。"""
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    x_cam = (us - K["cx"]) * zs / K["fx"]
    y_cam = sign_y * (vs - K["cy"]) * zs / K["fy"]
    cam = np.column_stack([x_cam, y_cam, -zs, np.ones_like(zs)])
    w = (np.linalg.inv(_V(view_matrix)) @ cam.T).T
    return w[:, :3] / w[:, 3:4]


def project_many(pts, view_matrix, proj_matrix):
    M = _V(proj_matrix) @ _V(view_matrix)
    h = np.column_stack([pts, np.ones(len(pts))])
    clip = (M @ h.T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    return (ndc[:, 0] + 1) / 2 * CAM_WIDTH, (1 - ndc[:, 1]) / 2 * CAM_HEIGHT


def depth_valid(depth):
    return (depth > CAM_NEAR + 0.005) & (depth < CAM_FAR - 0.005) & np.isfinite(depth)


def parse_K_from_proj(proj_matrix):
    """从 PyBullet 的投影矩阵解析真实内参（含可能存在的离轴 m02/m12）。"""
    P = _V(proj_matrix)
    m00, m02 = P[0, 0], P[0, 2]
    m11, m12 = P[1, 1], P[1, 2]
    return {
        "m00": m00, "m02": m02, "m11": m11, "m12": m12,
        "fx_px": CAM_WIDTH / 2.0 * m00,
        "fy_px": CAM_HEIGHT / 2.0 * m11,
        "cx_px": CAM_WIDTH / 2.0 * (1.0 + m02),
        "cy_px": CAM_HEIGHT / 2.0 * (1.0 - m12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="duck,teddy,cube")
    args = ap.parse_args()
    objects = [s.strip() for s in args.objects.split(",") if s.strip()]

    print("=" * 76)
    print("  🔬 Z 偏高根因诊断：内参 / 深度轴 / 融合口径")
    print("=" * 76)
    p.connect(p.DIRECT)
    _, robot_id, obj_infos = setup_simulation_multi(gui=False)
    init_panda_pose(robot_id)
    stabilize_objects(obj_infos, steps=240)

    gt = {}
    for info in obj_infos:
        aabb = p.getAABB(info["obj_id"])
        gt[info["name"]] = {"obj_id": info["obj_id"], "amin": list(aabb[0]),
                            "amax": list(aabb[1]),
                            "center": [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]}
    table_top = float(p.getAABB(1)[1][2])

    samples = []
    for nm, g in gt.items():
        if nm not in objects:
            continue
        tgt = [g["center"][0], g["center"][1], TABLE_SURFACE_Z]
        for vname, eye, target in ([("default", CAM_EYE, CAM_TARGET)]
                                   + [(f"V{i}", e, tgt) for i, e in enumerate(FUSION_EYES)]):
            rgb, depth, _, seg, vm, pm = get_camera_image(eye=eye, target=target)
            samples.append({"obj": nm, "view": vname, "eye": list(eye), "seg": seg,
                            "depth": depth, "vm": vm, "pm": pm})

    # ── 1. PyBullet 投影矩阵解析 ────────────────────────────────────
    print("\n【1】PyBullet 投影矩阵解析的真实内参")
    K_parsed = parse_K_from_proj(samples[0]["pm"])
    print(f"    m00={K_parsed['m00']:.6f}  m02={K_parsed['m02']:+.6f}   "
          f"m11={K_parsed['m11']:.6f}  m12={K_parsed['m12']:+.6f}")
    print(f"    → fx={K_parsed['fx_px']:.2f}px  fy={K_parsed['fy_px']:.2f}px  "
          f"cx={K_parsed['cx_px']:.2f}px  cy={K_parsed['cy_px']:.2f}px")
    print(f"    主线用  : f=(W/2)/tan(fov/2)={F_LEGACY:.2f}px  cx=320  cy=240+offset")
    print(f"    (H/2)/tan(fov/2)={F_VERT:.2f}px  ← 与解析值比较")

    # ── 2. 重投影自一致性 (u / v 分开看) ────────────────────────────
    print("\n【2】重投影自一致性：mask 像素 → 反投影 → PyBullet 矩阵再投影")
    print("     (f 错 → 残差 ∝ |像素-主点|；纯符号错 → 残差 = 2|像素-主点|)")

    def reproj(mask, depth, vm, pm, K, sign_y=1.0):
        m = mask & depth_valid(depth)
        if m.sum() < 50:
            return None
        ys, xs = np.where(m)
        ys, xs = ys.astype(np.float64), xs.astype(np.float64)
        pts = unproject(xs, ys, depth[ys.astype(int), xs.astype(int)], K, vm, sign_y)
        u2, v2 = project_many(pts, vm, pm)
        return (float(np.median(np.abs(u2 - xs))), float(np.median(np.abs(v2 - ys))),
                int(m.sum()), float(np.median(np.abs(xs - 320))), float(np.median(np.abs(ys - 240))))

    convs = [
        ("旧 f=554 cy=186, y+", dict(fx=F_LEGACY, fy=F_LEGACY, cx=320, cy=186), +1.0),
        ("旧 f=554 cy=186, y-", dict(fx=F_LEGACY, fy=F_LEGACY, cx=320, cy=186), -1.0),
        ("正确 f=416 cy=240, y+", dict(fx=F_VERT, fy=F_VERT, cx=320, cy=240), +1.0),
        ("正确 f=416 cy=240, y-", dict(fx=F_VERT, fy=F_VERT, cx=320, cy=240), -1.0),
    ]
    t2 = {}
    for label, K, sy in convs:
        dus, dvs = [], []
        for s in samples:
            mask = (s["seg"] == gt[s["obj"]]["obj_id"])
            r = reproj(mask, s["depth"], s["vm"], s["pm"], K, sy)
            if r:
                dus.append(r[0])
                dvs.append(r[1])
        t2[label] = {"du": round(float(np.median(dus)), 2),
                     "dv": round(float(np.median(dvs)), 2)}
        print(f"    {label:<22} 中位 |Δu|={t2[label]['du']:6.2f}px  |Δv|={t2[label]['dv']:6.2f}px")

    # ── 3. 物体点云 vs GT AABB（4 组约定）────────────────────────────
    print("\n【3】物体点云 vs GT AABB（同一 GT mask，只换反投影约定）")
    t3 = {}
    for label, K, sy in convs:
        print(f"\n    ▸ {label}")
        for nm, g in gt.items():
            if nm not in objects:
                continue
            per_view = []
            for s in samples:
                if s["obj"] != nm or s["view"] == "default":
                    continue
                mask = (s["seg"] == g["obj_id"]) & depth_valid(s["depth"])
                if mask.sum() < 100:
                    continue
                ys, xs = np.where(mask)
                pts = unproject(xs, ys, s["depth"][ys, xs], K, s["vm"], sy)
                per_view.append(pts)
            if not per_view:
                continue
            merged = np.vstack(per_view)
            cxy = merged[:, :2].mean(axis=0)
            zmin, zmax = float(merged[:, 2].min()), float(merged[:, 2].max())
            e_xy = float(np.linalg.norm(cxy - np.array(g["center"][:2]))) * 100
            d_lo = (zmin - g["amin"][2]) * 100
            d_hi = (zmax - g["amax"][2]) * 100
            print(f"      [{nm:<6}] 合并点云 XY误差={e_xy:5.2f}cm  Z=[{zmin:.3f},{zmax:.3f}]  "
                  f"GT AABB Z=[{g['amin'][2]:.3f},{g['amax'][2]:.3f}]  "
                  f"下界{d_lo:+6.1f}cm 上界{d_hi:+6.1f}cm")
            t3.setdefault(label, {})[nm] = {"xy_err_cm": round(e_xy, 2),
                                            "z_min": round(zmin, 4), "z_max": round(zmax, 4),
                                            "gt_z_min": round(g["amin"][2], 4),
                                            "gt_z_max": round(g["amax"][2], 4),
                                            "d_lo_cm": round(d_lo, 2), "d_hi_cm": round(d_hi, 2)}

    # ── 4. rayTest 命中率 ───────────────────────────────────────────
    print("\n【4】rayTest：mask 像素的相机射线能否命中该物体（正确约定应接近 100%）")
    t4 = {}
    for label, K, sy in convs:
        hit, tot = 0, 0
        errs = []
        for s in samples:
            oid = gt[s["obj"]]["obj_id"]
            mask = (s["seg"] == oid) & depth_valid(s["depth"])
            if mask.sum() < 50:
                continue
            ys, xs = np.where(mask)
            step = max(1, len(xs) // 25)
            xs, ys = xs[::step], ys[::step]
            Rcw = _V(s["vm"])[:3, :3]
            eye = np.array(s["eye"], dtype=np.float64)
            for u, v in zip(xs, ys):
                d_cam = np.array([(u - K["cx"]) / K["fx"], sy * (v - K["cy"]) / K["fy"], -1.0])
                d_world = Rcw.T @ (d_cam / np.linalg.norm(d_cam))
                hitres = p.rayTest(eye.tolist(), (eye + d_world * 2.9).tolist())[0]
                tot += 1
                if hitres[0] != oid:
                    continue
                hit += 1
                p_true = np.array(hitres[3])
                p_est = unproject(np.array([u]), np.array([v]),
                                  np.array([s["depth"][v, u]]), K, s["vm"], sy)[0]
                errs.append(np.linalg.norm(p_est - p_true) * 100)
        rate = 100.0 * hit / max(tot, 1)
        med = float(np.median(errs)) if errs else float("nan")
        t4[label] = {"hit_pct": round(rate, 1), "n_hit": hit, "n_total": tot,
                     "err3d_med_cm": None if not errs else round(med, 2)}
        print(f"    {label:<22} 命中率={rate:5.1f}% ({hit}/{tot})  "
              f"命中样本 3D 误差中位={med:.2f}cm" if errs
              else f"    {label:<22} 命中率={rate:5.1f}% ({hit}/{tot})")

    # ── 5. 修正后的逐视角一致性（融合口径是否还需要"补偿"）──────────
    print("\n【5】修正约定 (f=416, cy=240, y−) 下的逐视角点云（看融合口径是否还偏）")
    K_fix = dict(fx=F_VERT, fy=F_VERT, cx=320, cy=240)
    t5 = {}
    for nm, g in gt.items():
        if nm not in objects:
            continue
        print(f"    [{nm}] GT AABB Z=[{g['amin'][2]:.3f},{g['amax'][2]:.3f}] "
              f"XY中心=({g['center'][0]:.3f},{g['center'][1]:.3f})")
        zs, xs_, ys_ = [], [], []
        for s in samples:
            if s["obj"] != nm or s["view"] == "default":
                continue
            mask = (s["seg"] == g["obj_id"]) & depth_valid(s["depth"])
            if mask.sum() < 100:
                continue
            yy, xx = np.where(mask)
            pts = unproject(xx, yy, s["depth"][yy, xx], K_fix, s["vm"], -1.0)
            zim, zimax = float(np.median(pts[:, 2])), float(pts[:, 2].max())
            cxy = pts[:, :2].mean(axis=0)
            e_xy = float(np.linalg.norm(cxy - np.array(g["center"][:2]))) * 100
            zs.append(zim)
            xs_.append(cxy[0])
            ys_.append(cxy[1])
            print(f"        {s['view']:<4} n={mask.sum():>5} Z中位={zim:.3f} "
                  f"Zmax={zimax:.3f}  XY=({cxy[0]:.3f},{cxy[1]:.3f}) XY误差={e_xy:.2f}cm")
        if zs:
            t5[nm] = {"z_med_spread_cm": round((max(zs) - min(zs)) * 100, 2),
                      "z_med_mean": round(float(np.mean(zs)), 4),
                      "gt_mid": round(g["center"][2], 4),
                      "x_spread_cm": round((max(xs_) - min(xs_)) * 100, 2),
                      "y_spread_cm": round((max(ys_) - min(ys_)) * 100, 2),
                      "views": len(zs)}
            print(f"        → 视角间 Z中位 展布={t5[nm]['z_med_spread_cm']:.2f}cm  "
                  f"均值={t5[nm]['z_med_mean']:.3f} (GT中点 {t5[nm]['gt_mid']:.3f}, "
                  f"偏差 {(t5[nm]['z_med_mean'] - t5[nm]['gt_mid']) * 100:+.1f}cm)  "
                  f"X展布={t5[nm]['x_spread_cm']:.2f}cm Y展布={t5[nm]['y_spread_cm']:.2f}cm")

    runs = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    path = os.path.join(runs, f"z_bias_diag_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "parsed_intrinsics": K_parsed,
                   "legacy_F": F_LEGACY, "vert_F": F_VERT,
                   "table_top_z": table_top,
                   "T2_reproj": t2, "T3_cloud_vs_aabb": t3, "T4_raytest": t4, "T5_per_view_fixed": t5},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n💾 诊断明细 → {path}")
    p.disconnect()


if __name__ == "__main__":
    main()
