# scripts/generate_yolo_dataset.py
"""
生成自定义 YOLO 训练数据集。
从 PyBullet 仿真中采集图像 + 自动标注（使用 seg_img 提取 mask + bbox）。
"""

import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pybullet as p
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import (
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_WIDTH,
    MULTI_OBJECTS_CONFIG,
)
from src.env.sim_env import setup_simulation_multi

# ── 配置 ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo_dataset")
TRAIN_IMAGES = 150
VAL_IMAGES = 50
TABLE_Z = 0.625
MIN_OBJ_DIST = 0.15  # 物体间最小距离（防止重叠）
TABLE_X_MIN, TABLE_X_MAX = 0.20, 0.70
TABLE_Y_MIN, TABLE_Y_MAX = -0.40, 0.40

CLASS_NAMES = ["duck", "teddy", "cube"]  # 与 MULTI_OBJECTS_CONFIG 顺序一致

# 多组相机位姿（方位角 + 高度变化）
CAMERA_POSES = [
    {"eye": [1.3, -0.8, 1.3], "target": [0.45, 0.0, 0.65]},
    {"eye": [1.0, -0.6, 1.5], "target": [0.45, 0.0, 0.65]},
    {"eye": [0.6, -1.0, 1.2], "target": [0.45, 0.0, 0.65]},
    {"eye": [0.4, -1.1, 1.0], "target": [0.45, 0.0, 0.65]},
    {"eye": [-0.1, -0.6, 1.3], "target": [0.45, 0.0, 0.65]},
    {"eye": [-0.2, -0.4, 1.1], "target": [0.45, 0.0, 0.65]},
    {"eye": [1.4, -0.2, 1.2], "target": [0.45, 0.0, 0.65]},
    {"eye": [1.3, 0.0, 1.0], "target": [0.45, 0.0, 0.65]},
    {"eye": [0.0, -0.2, 1.6], "target": [0.45, 0.0, 0.65]},
    {"eye": [0.8, -1.2, 1.1], "target": [0.45, 0.0, 0.65]},
]


def _random_nonoverlap_position(existing_positions, rng):
    """
    在桌面范围内生成不与其他物体重叠的随机位置。
    existing_positions: list of [x, y, z]
    returns: [x, y, z] or None (如果桌面已满)
    """
    for _attempt in range(50):
        x = rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
        y = rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)
        pos = [x, y, TABLE_Z + 0.02]
        # 检查与已有物体的距离
        if all(
            math.hypot(x - ex, y - ey) >= MIN_OBJ_DIST
            for ex, ey, _ in existing_positions
        ):
            return pos
    return None  # 尝试 50 次都找不到合适位置


def randomize_objects(obj_infos, rng):
    """随机化物体位置（不重叠），返回位置列表。"""
    placed = []
    for info in obj_infos:
        pos = _random_nonoverlap_position(placed, rng)
        if pos is None:
            # 桌面太挤，fallback 到固定间距
            idx = len(placed)
            x = TABLE_X_MIN + 0.1 + idx * 0.15
            y = TABLE_Y_MIN + 0.1 + (idx % 3) * 0.15
            pos = [min(x, TABLE_X_MAX), min(y, TABLE_Y_MAX), TABLE_Z + 0.02]

        yaw = rng.uniform(0, 360)
        if info["name"] == "duck":
            quat = p.getQuaternionFromEuler([math.pi / 2, 0, math.radians(yaw)])
        else:
            quat = p.getQuaternionFromEuler([0, 0, math.radians(yaw)])

        p.resetBasePositionAndOrientation(info["obj_id"], pos, quat)
        p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])
        placed.append(pos)

    for _ in range(120):
        p.stepSimulation()

    return placed


def mask_to_polygon(mask, epsilon=1.0):
    """二值 mask → 归一化多边形顶点列表 [(x,y), ...]。"""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None
    approx = cv2.approxPolyDP(largest, epsilon, True)
    h, w = mask.shape
    return [(float(pt[0][0]) / w, float(pt[0][1]) / h) for pt in approx]


def capture_and_label(obj_infos, cam_pose, img_idx, split):
    """采集一张图并生成 YOLO 格式标注。"""
    eye = cam_pose["eye"]
    target = cam_pose["target"]
    up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(eye, target, up)
    proj_matrix = p.computeProjectionMatrixFOV(
        CAM_FOV_DEG, CAM_WIDTH / CAM_HEIGHT, 0.1, 3.0
    )

    _, _, rgb_raw, _depth_buf, seg_raw = p.getCameraImage(
        CAM_WIDTH, CAM_HEIGHT, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgb = np.reshape(rgb_raw, (CAM_HEIGHT, CAM_WIDTH, 4)).astype(np.uint8)[:, :, :3]
    seg_img = np.reshape(seg_raw, (CAM_HEIGHT, CAM_WIDTH)).astype(np.int32)

    # 保存图片
    img_dir = os.path.join(OUTPUT_DIR, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"img_{img_idx:06d}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # 生成标注（YOLO seg 格式）
    labels_dir = os.path.join(OUTPUT_DIR, "labels", split)
    os.makedirs(labels_dir, exist_ok=True)
    label_path = os.path.join(labels_dir, f"img_{img_idx:06d}.txt")

    annotations = []
    for info in obj_infos:
        class_id = next(i for i, c in enumerate(CLASS_NAMES) if c == info["name"])
        mask = (seg_img == info["obj_id"]).astype(np.uint8)
        if np.sum(mask) < 50:
            continue
        polygon = mask_to_polygon(mask)
        if polygon is None or len(polygon) < 3:
            continue
        line = f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        annotations.append(line)

    with open(label_path, "w") as f:
        f.write("\n".join(annotations))

    return img_path, len(annotations)


def count_objects_in_label(label_path):
    """读取标注文件，返回 {class_name: count}。"""
    counts = {}
    if not os.path.exists(label_path):
        return counts
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cls_id = int(line.split()[0])
                if cls_id < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_id]
                    counts[name] = counts.get(name, 0) + 1
    return counts


def main():
    print("=" * 60)
    print("  🏋️  生成 YOLO 训练数据集")
    print("=" * 60)

    # ── 1. 启动仿真 ──────────────────────────────────────────────────
    print("\n📦 启动仿真环境（DIRECT 模式）...")
    client, robot_id, obj_infos = setup_simulation_multi(gui=False)
    print(f"   共 {len(obj_infos)} 个物体")

    # ── 2. 生成数据集 ────────────────────────────────────────────────
    total = TRAIN_IMAGES + VAL_IMAGES
    print(f"\n📸 采集 {total} 张图片（训练 {TRAIN_IMAGES} + 验证 {VAL_IMAGES}）...")

    rng = random.Random(42)
    stats = {"train": {"total": 0, "with_objects": 0},
             "val": {"total": 0, "with_objects": 0}}
    obj_stats = {"train": {name: 0 for name in CLASS_NAMES},
                 "val": {name: 0 for name in CLASS_NAMES}}

    for i in range(total):
        split = "train" if i < TRAIN_IMAGES else "val"

        randomize_objects(obj_infos, rng)

        cam_pose = rng.choice(CAMERA_POSES)
        cam_pose = {
            "eye": [e + rng.uniform(-0.05, 0.05) for e in cam_pose["eye"]],
            "target": cam_pose["target"],
        }

        img_path, visible_count = capture_and_label(obj_infos, cam_pose, i, split)

        stats[split]["total"] += 1
        if visible_count > 0:
            stats[split]["with_objects"] += 1

        # 统计类别（用正确路径构建）
        label_path = os.path.join(
            OUTPUT_DIR, "labels", split, f"img_{i:06d}.txt"
        )
        counts = count_objects_in_label(label_path)
        for name, cnt in counts.items():
            obj_stats[split][name] += cnt

        if (i + 1) % 50 == 0 or i == 0:
            print(f"   进度: {i + 1}/{total}  ({split}: img_{i:06d}, {visible_count} objects)")

    # ── 3. 生成 dataset.yaml ─────────────────────────────────────────
    data_yaml = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # ── 4. 报告 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 数据集生成完成")
    print("=" * 60)
    print(f"\n  📁 {OUTPUT_DIR}")
    print(f"\n  训练集: {stats['train']['total']} 张 ({stats['train']['with_objects']} 张有物体)")
    print(f"  验证集: {stats['val']['total']} 张 ({stats['val']['with_objects']} 张有物体)")
    print(f"\n  各类别出现次数:")
    for split_name in ["train", "val"]:
        print(f"    [{split_name}]")
        for name in CLASS_NAMES:
            print(f"      {name}: {obj_stats[split_name][name]}")

    p.disconnect()
    print("\n👋 完成")


if __name__ == "__main__":
    main()
