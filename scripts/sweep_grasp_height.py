# scripts/sweep_grasp_height.py
"""
抓取高度扫描：重标 waist_ratio 用（内参修正 + 闭环微调之后）。

为什么需要
----------
内参修正前，点云 Z 偏高 ~10cm，"grasp_z" 只对某一组凑合参数有效：
`ik_bias` + `min_fine_z` 钳制 + IK 稳态残差三者叠加，实际指尖落点比"命令腰线"低 2~4cm，
而物体恰好能被那种（贴近桌面的）位置夹住 —— 所以旧 waist_ratio 从未真正验证过。

闭环微调后命令=实际（Z 残差 ≤2mm），因此必须重新用实验确定"每个物体真实的合理夹取高度"。

用法
----
    uv run python scripts/sweep_grasp_height.py --heights 0.60,0.62,0.64,0.66
    uv run python scripts/sweep_grasp_height.py --objects teddy,cube --heights 0.62,0.64

每个 (物体, 高度) 独立重建仿真与物体初始位姿；成功判据与主线一致（轻提 Δz>1cm）。
输出：控制台成功矩阵 + runs/sweep_grasp_height_<时间戳>.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pybullet as p  # noqa: E402

from src.configs.config import MULTI_OBJECTS_CONFIG, TABLE_SURFACE_Z  # noqa: E402
from src.env.sim_env import init_panda_pose, setup_simulation_multi, stabilize_objects  # noqa: E402
from src.perception.detector import YoloSegmentor  # noqa: E402


def _load_main():
    path = os.path.join(REPO_ROOT, "scripts", "test_multi_object_grasp.py")
    spec = importlib.util.spec_from_file_location("grasp_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grasp_main"] = mod
    spec.loader.exec_module(mod)
    return mod


MAIN = _load_main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="duck,teddy,cube")
    ap.add_argument("--heights", default="0.60,0.62,0.64,0.66",
                    help="指尖目标高度（m，逗号分隔）")
    ap.add_argument("--gui", action="store_true", default=True)
    args = ap.parse_args()
    objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    heights = [float(s) for s in args.heights.split(",") if s.strip()]

    model_path = os.path.join(REPO_ROOT, "models", "custom_yolov8n_seg.pt")
    print("=" * 78)
    print("  📐 抓取高度扫描（内参修正 + 闭环微调后的重标）")
    print(f"  物体={objects}  指尖目标高度={heights}")
    print(f"  桌面 z={TABLE_SURFACE_Z}  成功判据 Δz>1cm")
    print("=" * 78)
    det = YoloSegmentor(model_path=model_path)

    rows = []
    for h in heights:
        print(f"\n{'─' * 78}\n  指尖目标高度 = {h:.3f} m\n{'─' * 78}")
        for name in objects:
            client, robot_id, obj_infos = setup_simulation_multi(gui=args.gui)
            init_panda_pose(robot_id)
            stabilize_objects(obj_infos, steps=240)
            info = next(i for i in obj_infos if i["name"] == name)
            aabb = p.getAABB(info["obj_id"])
            gt_xy = [(aabb[0][0] + aabb[1][0]) / 2, (aabb[0][1] + aabb[1][1]) / 2]

            pos, gp = MAIN.estimate_object_pose(name, MAIN.YOLO_CLASS_IDS[name],
                                                [0.45, 0.0, 0.65], det)
            if pos is None:
                print(f"  [{name}] ❌ 位姿估计失败")
                rows.append({"object": name, "target_z": h, "success": False,
                             "error": "pose_failed"})
                p.disconnect()
                continue

            gp = dict(gp)
            gp["grasp_z"] = h          # 覆盖腰线高度，其余沿用主线参数
            t0 = time.time()
            try:
                ok = bool(MAIN.grasp_object(robot_id, info["obj_id"], name, pos, gp))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] ❌ 抓取异常: {exc}")
                ok = False
            achieved = MAIN.get_ee_pos(robot_id)[2] if ok else None
            print(f"  [{name}] 目标指尖={h:.3f} 实际 hand_Z={achieved if achieved is None else round(achieved,3)} "
                  f"→ {'✅ 成功' if ok else '❌ 失败'}  ({time.time()-t0:.0f}s)")
            rows.append({"object": name, "target_z": h, "success": ok,
                         "gt_xy": [round(gt_xy[0], 4), round(gt_xy[1], 4)],
                         "aabb_z": [round(aabb[0][2], 4), round(aabb[1][2], 4)],
                         "est_xy": [round(float(pos[0]), 4), round(float(pos[1]), 4)],
                         "z_range": round(float(gp["z_range"]), 4),
                         "achieved_hand_z": None if achieved is None else round(float(achieved), 4)})
            p.disconnect()

    print(f"\n{'=' * 78}\n  📊 成功矩阵（行=物体，列=指尖目标高度）\n{'=' * 78}")
    hdr = "  " + f"{'物体':<8}" + "".join(f"{h:>10.3f}" for h in heights)
    print(hdr)
    for name in objects:
        cells = []
        for h in heights:
            hit = [r for r in rows if r["object"] == name and abs(r["target_z"] - h) < 1e-9]
            cells.append("✅" if hit and hit[0]["success"] else ("❌" if hit else "-"))
        print("  " + f"{name:<8}" + "".join(f"{c:>10}" for c in cells))

    runs = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    path = os.path.join(runs, f"sweep_grasp_height_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"heights": heights, "objects": objects, "rows": rows,
                   "table_z": TABLE_SURFACE_Z,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n💾 明细 → {path}")


if __name__ == "__main__":
    main()
