# scripts/batch_test_final.py
"""
批量统计：多视角融合定位精度 + 多物体抓取成功率。

STEP12 重写说明
--------------
旧版本脚本自带一套与主线相反的参数（cy_offset=-6、PCA 主轴算 yaw、
z_min + 0.35*z_range 抓取高度、force=800/500），这些结论已被 STEP10 推翻
（PCA 不可行 → 类硬编码 yaw；抓取高度 → 桌面基准腰线）。为杜绝"测试脚本与
主线不一致"的漂移，本版**不复制任何估计/抓取逻辑**，直接 import
`scripts/test_multi_object_grasp.py` 并调用它的 `estimate_object_pose()` /
`grasp_object()`；主线改一次，这里自动跟随。

两阶段
------
Phase A 定位统计（DIRECT 渲染，快）:
    每个物体每轮 → estimate_object_pose() 的 3D 估计 vs PyBullet GT
    （GT 仅用于打分，不参与任何估计/抓取决策）→ 记录 XY 误差、Z 偏差
Phase B 抓取统计（GUI，慢）:
    成功率；判据与主线一致（轻提后物体 Δz > 1cm）

用法
----
    uv run python scripts/batch_test_final.py --pose-trials 5 --grasp-trials 3
    uv run python scripts/batch_test_final.py --pose-trials 5 --grasp-trials 0   # 只做定位
    uv run python scripts/batch_test_final.py --grasp-trials 3 --jitter 0.02     # 位置抖动鲁棒性

输出
----
    控制台汇总表 + runs/batch_stats_<时间戳>.json（runs/ 已在 .gitignore 中）
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pybullet as p  # noqa: E402

from src.configs.config import MULTI_OBJECTS_CONFIG  # noqa: E402
from src.env.sim_env import (  # noqa: E402
    init_panda_pose,
    setup_simulation_multi,
    stabilize_objects,
)
from src.perception.detector import YoloSegmentor  # noqa: E402


def _load_main_module():
    """按文件路径载入主线脚本（复用其估计/抓取函数，避免逻辑重复）。"""
    path = os.path.join(REPO_ROOT, "scripts", "test_multi_object_grasp.py")
    spec = importlib.util.spec_from_file_location("grasp_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grasp_main"] = mod
    spec.loader.exec_module(mod)
    return mod


MAIN = _load_main_module()

MODEL_PATH = os.path.join(REPO_ROOT, "models", "custom_yolov8n_seg.pt")
CAM_TARGET_ROUGH = [0.45, 0.0, 0.65]   # 与主线一致的固定相机指向
ALL_OBJECTS = ["duck", "teddy", "cube"]


def build_objects_config(jitter, rng):
    """按 config.MULTI_OBJECTS_CONFIG 构造本轮物体配置，可选 ±jitter 位置抖动。"""
    cfgs = []
    for cfg in MULTI_OBJECTS_CONFIG:
        c = dict(cfg)
        c["pos"] = list(cfg["pos"])
        if jitter > 0:
            c["pos"][0] += float(rng.uniform(-jitter, jitter))
            c["pos"][1] += float(rng.uniform(-jitter, jitter))
        cfgs.append(c)
    return cfgs


def _spawn(gui, jitter, rng):
    cfgs = build_objects_config(jitter, rng)
    client, robot_id, obj_infos = setup_simulation_multi(gui=gui, objects_config=cfgs)
    init_panda_pose(robot_id)
    stabilize_objects(obj_infos, steps=240)
    return client, robot_id, obj_infos


# ── Phase A: 定位精度 ────────────────────────────────────────────────
def phase_pose(trials, objects, jitter, rng, detector):
    print("\n" + "=" * 64)
    print(f"  📐 Phase A: 多视角融合定位精度 ({trials} 轮, DIRECT 渲染)")
    print("=" * 64)

    rows = []
    for t in range(trials):
        client, robot_id, obj_infos = _spawn(gui=False, jitter=jitter, rng=rng)
        print(f"\n── trial {t + 1}/{trials} ──")
        for info in obj_infos:
            name = info["name"]
            if name not in objects:
                continue
            gt = list(p.getBasePositionAndOrientation(info["obj_id"])[0])
            # 评分基准：点云質心是"可见表面"的质心，应与物体实际几何范围比，
            # 而不是与 URDF 基座原点比（两者在 Z 上天然差半个物体高度）。
            aabb = p.getAABB(info["obj_id"])
            gt_xy = np.array([(aabb[0][0] + aabb[1][0]) / 2, (aabb[0][1] + aabb[1][1]) / 2])
            gt_z_mid = (aabb[0][2] + aabb[1][2]) / 2
            est, _ = MAIN.estimate_object_pose(
                name, MAIN.YOLO_CLASS_IDS[name], CAM_TARGET_ROUGH, detector
            )
            if est is None:
                print(f"   [{name}] ❌ 估计失败")
                rows.append({"trial": t, "object": name, "ok": False})
                continue
            xy_err = float(np.linalg.norm(np.array(est[:2]) - gt_xy))
            z_err = float(est[2] - gt_z_mid)
            err3d = float(np.sqrt(xy_err ** 2 + z_err ** 2))
            z_err_base = float(est[2] - gt[2])
            print(f"   [{name}] est=({est[0]:.3f},{est[1]:.3f},{est[2]:.3f}) "
                  f"gt_xy=({gt_xy[0]:.3f},{gt_xy[1]:.3f}) gt_z范围=[{aabb[0][2]:.3f},{aabb[1][2]:.3f}] | "
                  f"XY={xy_err * 100:.1f}cm  Z={z_err * 100:+.1f}cm(vs中点) "
                  f"{z_err_base * 100:+.1f}cm(vs基座)  3D={err3d * 100:.1f}cm")
            rows.append({
                "trial": t, "object": name, "ok": True,
                "est": [round(float(v), 4) for v in est],
                "gt": [round(float(v), 4) for v in gt],
                "gt_xy": [round(float(gt_xy[0]), 4), round(float(gt_xy[1]), 4)],
                "gt_z_range": [round(float(aabb[0][2]), 4), round(float(aabb[1][2]), 4)],
                "xy_err_cm": round(xy_err * 100, 2),
                "z_err_cm": round(z_err * 100, 2),
                "z_err_base_cm": round(z_err_base * 100, 2),
                "err3d_cm": round(err3d * 100, 2),
            })
        p.disconnect()
    return rows


# ── Phase B: 抓取成功率 ──────────────────────────────────────────────
def phase_grasp(trials, objects, jitter, rng, detector):
    print("\n" + "=" * 64)
    print(f"  🤖 Phase B: 多物体抓取成功率 ({trials} 轮, GUI, 成功判据 Δz>1cm)")
    print("=" * 64)

    rows = []
    for t in range(trials):
        client, robot_id, obj_infos = _spawn(gui=True, jitter=jitter, rng=rng)
        print(f"\n── trial {t + 1}/{trials} ──")
        for idx, info in enumerate(obj_infos):
            name = info["name"]
            if name not in objects:
                continue
            if idx > 0:
                init_panda_pose(robot_id)
                MAIN.wait(steps=60)

            est, grasp_params = MAIN.estimate_object_pose(
                name, MAIN.YOLO_CLASS_IDS[name], CAM_TARGET_ROUGH, detector
            )
            if est is None:
                print(f"   [{name}] ❌ 位姿估计失败")
                rows.append({"trial": t, "object": name, "success": False,
                             "error": "pose_estimation_failed"})
                continue

            try:
                ok = bool(MAIN.grasp_object(robot_id, info["obj_id"], name, est, grasp_params))
            except Exception as exc:  # noqa: BLE001
                print(f"   [{name}] ❌ 抓取异常: {exc}")
                ok = False
                if "Not connected" in str(exc):
                    rows.append({"trial": t, "object": name, "success": False,
                                 "error": "simulation_disconnected"})
                    print("   ⚠️  PyBullet 断开，终止本轮")
                    break
            rows.append({"trial": t, "object": name, "success": ok})
        try:
            p.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return rows


def summarize(pose_rows, grasp_rows, objects):
    print("\n" + "=" * 64)
    print("  📊 批量统计汇总")
    print("=" * 64)

    summary = {"pose": {}, "grasp": {}}

    if pose_rows:
        print("\n[定位精度]  物体        XY误差(cm)          Z偏差(cm,vs AABB中点)  3D误差(cm)   N")
        for name in objects:
            rs = [r for r in pose_rows if r["object"] == name and r.get("ok")]
            if not rs:
                continue
            xy = [r["xy_err_cm"] for r in rs]
            zs = [r["z_err_cm"] for r in rs]
            e3 = [r["err3d_cm"] for r in rs]
            f = lambda v: f"{statistics.mean(v):.2f}±{statistics.pstdev(v):.2f}"  # noqa: E731
            print(f"              {name:<8} {f(xy):<18} {f(zs):<17} {f(e3):<11} {len(rs)}")
            summary["pose"][name] = {
                "n": len(rs),
                "xy_err_cm_mean": round(statistics.mean(xy), 3),
                "xy_err_cm_std": round(statistics.pstdev(xy), 3),
                "z_err_cm_mean": round(statistics.mean(zs), 3),
                "z_err_cm_std": round(statistics.pstdev(zs), 3),
                "err3d_cm_mean": round(statistics.mean(e3), 3),
            }
        failed = [r for r in pose_rows if not r.get("ok")]
        if failed:
            print(f"              ⚠️ 估计失败 {len(failed)} 次")

    if grasp_rows:
        print("\n[抓取成功率]")
        total_ok = total = 0
        for name in objects:
            rs = [r for r in grasp_rows if r["object"] == name]
            if not rs:
                continue
            ok = sum(1 for r in rs if r["success"])
            total_ok += ok
            total += len(rs)
            print(f"              {name:<8} {ok}/{len(rs)}  ({ok / len(rs) * 100:.0f}%)")
            summary["grasp"][name] = {"success": ok, "n": len(rs),
                                      "rate": round(ok / len(rs), 3)}
        if total:
            print(f"              {'总计':<8} {total_ok}/{total}  ({total_ok / total * 100:.0f}%)")
            summary["grasp"]["overall"] = {"success": total_ok, "n": total,
                                           "rate": round(total_ok / total, 3)}
    return summary


def main():
    ap = argparse.ArgumentParser(description="多视角定位精度 + 抓取成功率批量统计")
    ap.add_argument("--pose-trials", type=int, default=5, help="Phase A 轮数（DIRECT）")
    ap.add_argument("--grasp-trials", type=int, default=3, help="Phase B 轮数（GUI）")
    ap.add_argument("--objects", default=",".join(ALL_OBJECTS), help="逗号分隔物体名")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="物体初始位置 ±抖动（米），0=固定复现配置")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    print("=" * 64)
    print("  📊 批量统计: 定位精度 + 抓取成功率")
    print(f"  物体={objects}  pose-trials={args.pose_trials}  "
          f"grasp-trials={args.grasp_trials}  jitter=±{args.jitter}m  seed={args.seed}")
    print("=" * 64)

    t0 = time.time()
    print(f"\n🤖 加载 YOLO 模型 ({os.path.relpath(MODEL_PATH, REPO_ROOT)})...")
    detector = YoloSegmentor(model_path=MODEL_PATH)

    pose_rows = []
    if args.pose_trials > 0:
        pose_rows = phase_pose(args.pose_trials, objects, args.jitter, rng, detector)

    grasp_rows = []
    if args.grasp_trials > 0:
        grasp_rows = phase_grasp(args.grasp_trials, objects, args.jitter, rng, detector)

    summary = summarize(pose_rows, grasp_rows, objects)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "objects": objects,
            "pose_trials": args.pose_trials,
            "grasp_trials": args.grasp_trials,
            "jitter_m": args.jitter,
            "seed": args.seed,
            "model": os.path.relpath(MODEL_PATH, REPO_ROOT),
        },
        "summary": summary,
        "pose_rows": pose_rows,
        "grasp_rows": grasp_rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    runs_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"batch_stats_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"\n⏱️  总耗时 {out['wall_seconds']}s")
    print(f"💾 明细 → {out_path}")
    print("\n👋 完成")


if __name__ == "__main__":
    main()
