# scripts/test_multi_object_detection.py
"""
第二步：多物体检测 + 位姿估计。
使用 YOLOv8-seg 检测仿真环境中的多个物体，
同时用 PyBullet 分割图验证，估算每个物体的 3D 世界坐标。
"""

import os
import sys
import time

import cv2
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import CAM_EYE, CAM_HEIGHT, CAM_TARGET, CAM_UP, CAM_WIDTH
from src.env.sim_env import setup_simulation_multi, init_panda_pose, stabilize_objects
from src.perception.camera import get_camera_image
from src.perception.detector import YoloSegmentor
from src.perception.pose_estimator import PoseEstimator

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def estimate_3d_from_mask(depth, binary_mask, view_matrix, estimator):
    """
    用二进制 mask 估算物体 3D 位置（世界坐标）。
    创建一个仅含该 mask 的合成 seg_img，然后交给 PoseEstimator 处理。
    """
    h, w = binary_mask.shape
    fake_seg = np.zeros((h, w), dtype=np.int32)
    MASK_OBJ_ID = -1  # 使用负数避免与真实 obj_id 冲突
    fake_seg[binary_mask > 0] = MASK_OBJ_ID

    result = estimator.estimate(
        depth=depth,
        seg_img=fake_seg,
        view_matrix=view_matrix,
        obj_id=MASK_OBJ_ID,
    )
    return result


def draw_detection(image, box, position_world, label, color):
    """在图像上画检测框 + 3D 位置文字。"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label}"
    if position_world is not None:
        text += f" ({position_world[0]:.2f}, {position_world[1]:.2f}, {position_world[2]:.2f})"
    cv2.putText(image, text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def main():
    print("=" * 60)
    print("  🚀 第二步：多物体检测 + 位姿估计")
    print("=" * 60)

    # ── 1. 加载多物体环境 ──────────────────────────────────────────────
    print("\n📦 加载多物体环境...")
    client, robot_id, obj_infos = setup_simulation_multi(gui=True)

    # 初始化机械臂姿态，让出视野
    init_panda_pose(robot_id)

    # 稳定物体
    stabilize_objects(obj_infos, steps=240)

    print(f"   已加载 {len(obj_infos)} 个物体")
    for info in obj_infos:
        print(f"     [{info['name']}] obj_id={info['obj_id']} @ {info['pos']}")

    # ── 2. 相机采集 ───────────────────────────────────────────────────
    print("\n📸 采集相机数据（RGB + 深度 + 分割）...")
    rgb_bgr, depth_real, depth_vis, seg_img, view_matrix, proj_matrix = (
        get_camera_image()
    )
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # 保存原始图
    cv2.imwrite(os.path.join(ASSETS_DIR, "m2_rgb.png"), rgb_bgr)
    cv2.imwrite(os.path.join(ASSETS_DIR, "m2_depth.png"), depth_vis)
    print(f"   已保存 RGB / 深度图到 assets/")

    # ── 3. 自标定 cy_offset + PyBullet 分割图位姿估计 ────────────────
    print("\n🔧 单视角自标定 cy_offset（桌面约束）...")

    TABLE_SURFACE_Z = 0.625
    # 搜索使各物体点云底部接近桌面的 cy_offset
    best_offset = 0
    best_score = float("inf")

    for offset in range(-120, 20, 2):
        est = PoseEstimator(cy_offset=offset, verbose=False)
        z_errors = []
        for info in obj_infos:
            stats = est.get_point_cloud_stats(depth_real, seg_img, view_matrix, info["obj_id"])
            if stats and stats["num_points"] >= 30:
                # 点云底部应接近桌面高度
                z_errors.append(abs(stats["z_min"] - TABLE_SURFACE_Z))
        if z_errors:
            score = float(np.mean(z_errors))
            if score < best_score:
                best_score = score
                best_offset = offset

    print(f"   自标定: cy_offset={best_offset:+d}, 桌面偏差={best_score * 100:.1f}cm")

    estimator = PoseEstimator(cy_offset=best_offset, verbose=False)
    print(f"\n🔍 [方法 A] 利用 PyBullet 分割图逐物体位姿估计...")
    seg_results = []
    for info in obj_infos:
        obj_id = info["obj_id"]
        mask_pixels = np.sum(seg_img == obj_id)
        if mask_pixels < 50:
            print(f"   ⚠️ [{info['name']}] 在分割图中像素太少 ({mask_pixels})，跳过")
            continue

        result = estimator.estimate(
            depth=depth_real,
            seg_img=seg_img,
            view_matrix=view_matrix,
            obj_id=obj_id,
        )
        if result is not None:
            gt_pos = p.getBasePositionAndOrientation(obj_id)[0]
            err = np.linalg.norm(result["position_world"] - np.array(gt_pos))
            seg_results.append({
                "name": info["name"],
                "obj_id": obj_id,
                "position_world": result["position_world"],
                "gt_pos": np.array(gt_pos),
                "error_cm": err * 100,
                "num_points": result["num_points"],
            })
            print(f"   ✅ [{info['name']}] pos=({result['position_world'][0]:.3f}, "
                  f"{result['position_world'][1]:.3f}, {result['position_world'][2]:.3f}) "
                  f"GT误差={err*100:.1f}cm  ({result['num_points']} 点)")
        else:
            print(f"   ❌ [{info['name']}] 位姿估计失败")

    # ── 4. YOLO 检测（自定义模型） ──────────────────────────────────
    print("\n🤖 [方法 B] 自定义 YOLOv8-seg 检测...")
    custom_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "models", "custom_yolov8n_seg.pt")
    if os.path.exists(custom_model_path):
        detector = YoloSegmentor(model_path=custom_model_path)
    else:
        print("   ⚠️ 自定义模型不存在，使用预训练 COCO 模型")
        detector = YoloSegmentor()
    yolo_detections = detector.detect_all(rgb)

    yolo_results = []
    for i, det in enumerate(yolo_detections):
        cls_id = det["cls_id"]
        class_name = det["class_name"]  # 来自模型自身 names 映射
        conf = det["conf"]
        box = det["box"]

        # YOLO mask 方式估算 3D 位置
        pos = None
        if det["mask"] is not None:
            mask = cv2.resize(det["mask"], (CAM_WIDTH, CAM_HEIGHT))
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_px = np.sum(mask_bin)
            if mask_px >= 50:
                result = estimate_3d_from_mask(depth_real, mask_bin, view_matrix, estimator)
                if result is not None:
                    pos = result["position_world"]

        yolo_results.append({
            "idx": i,
            "class_name": class_name,
            "conf": conf,
            "box": box,
            "position_world": pos,
        })
        pos_str = f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})" if pos is not None else "N/A"
        print(f"   {i}: {class_name} conf={conf:.2f} box=({int(box[0])},{int(box[1])}) "
              f"pos={pos_str}")

    if not yolo_detections:
        print("   ⚠️ YOLO 未检测到任何物体")

    # ── 5. 可视化 ─────────────────────────────────────────────────────
    print("\n🎨 生成可视化...")
    vis = rgb_bgr.copy()

    # 画 PyBullet 分割图结果（绿色框）
    colors = [(0, 200, 0), (0, 200, 200), (200, 0, 200)]
    for i, sr in enumerate(seg_results):
        # 从射影几何反推 2D 框：用 mask 的 bbox
        obj_mask = (seg_img == sr["obj_id"])
        ys, xs = np.where(obj_mask)
        if len(xs) > 0:
            box_2d = [xs.min(), ys.min(), xs.max(), ys.max()]
            color = colors[i % len(colors)]
            label = f"{sr['name']} (seg)"
            if sr['error_cm'] < 10:
                label += f" err={sr['error_cm']:.1f}cm"
            draw_detection(vis, box_2d, sr["position_world"], label, color)

    # 画 YOLO 检测结果（红色框）
    for yr in yolo_results:
        box = yr["box"]
        draw_detection(vis, box, yr["position_world"],
                       f"YOLO:{yr['class_name']}({yr['conf']:.2f})",
                       (0, 0, 200))

    save_path = os.path.join(ASSETS_DIR, "m2_detection_result.png")
    cv2.imwrite(save_path, vis)
    print(f"   可视化已保存 → {save_path}")

    # ── 6. 汇总报告 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 检测汇总")
    print("=" * 60)
    print(f"\n  [分割图] 共检测 {len(seg_results)}/{len(obj_infos)} 个物体:")
    for sr in seg_results:
        err_str = f"误差={sr['error_cm']:.1f}cm" if sr['error_cm'] < 10 else f"⚠️ 误差={sr['error_cm']:.1f}cm"
        print(f"    {sr['name']:>8s}: ({sr['position_world'][0]:.3f}, {sr['position_world'][1]:.3f}, {sr['position_world'][2]:.3f})  {err_str}")

    print(f"\n  [YOLO] 共检测 {len(yolo_detections)} 个物体")
    for yr in yolo_results:
        print(f"    {yr['class_name']:>12s} conf={yr['conf']:.2f}  pos={'✓' if yr['position_world'] is not None else '✗'}")

    print("\n✅ 第二步完成！")
    print(f"   可视化 → assets/m2_detection_result.png")

    # 保持 3 秒查看
    for _ in range(720):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
    print("👋 退出")


if __name__ == "__main__":
    main()
