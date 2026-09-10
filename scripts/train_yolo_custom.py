# scripts/train_yolo_custom.py
"""
训练自定义 YOLOv8-seg 模型，识别仿真中的 duck/teddy/cube。

使用 generate_yolo_dataset.py 生成的数据集。
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ── 配置 ─────────────────────────────────────────────────────────────
DATASET_YAML = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "yolo_dataset", "dataset.yaml")
MODEL_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models")
EPOCHS = 80
IMGSZ = 640
BATCH = 8
PATIENCE = 15


def main():
    print("=" * 60)
    print("  🏋️  自定义 YOLOv8-seg 训练")
    print("=" * 60)

    # 检查数据集
    if not os.path.exists(DATASET_YAML):
        print(f"❌ 数据集不存在: {DATASET_YAML}")
        print("   请先运行 scripts/generate_yolo_dataset.py")
        return

    # 创建输出目录
    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📦 设备: {device}")
    print(f"   数据集: {DATASET_YAML}")
    print(f"   轮数: {EPOCHS}")
    print(f"   图片尺寸: {IMGSZ}")
    if device == "cpu":
        print("   ⚠️  使用 CPU 训练，速度较慢（预计 15-30 分钟）")

    # ── 加载模型 ─────────────────────────────────────────────────────
    from ultralytics import YOLO

    print("\n⏳ 加载预训练 YOLOv8n-seg 模型...")
    model = YOLO("yolov8n-seg.pt")
    print("✅ 模型加载完成")

    # ── 训练 ─────────────────────────────────────────────────────────
    print("\n🏃 开始训练...")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        patience=PATIENCE,
        project=MODEL_OUTPUT,
        name="yolo_duck_teddy_cube",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        augment=True,
        # 对小数据集有利的参数
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,        # 随机旋转
        translate=0.1,       # 随机平移
        scale=0.5,           # 随机缩放
        shear=2.0,           # 随机剪切
        perspective=0.0,
        flipud=0.0,          # 不上下翻转（桌面场景不适用）
        fliplr=0.5,          # 左右翻转
        mosaic=0.5,          # mosaic 增强
        mixup=0.1,           # mixup 增强
        copy_paste=0.1,      # copy-paste 增强
    )

    # ── 验证 ─────────────────────────────────────────────────────────
    print("\n📊 在验证集上评估...")
    metrics = model.val()

    print(f"\n   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Mask mAP50: {metrics.seg.map50:.4f}")

    # ── 导出 ─────────────────────────────────────────────────────────
    best_path = os.path.join(MODEL_OUTPUT, "yolo_duck_teddy_cube", "weights", "best.pt")
    final_path = os.path.join(MODEL_OUTPUT, "custom_yolov8n_seg.pt")
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, final_path)
        print(f"\n📦 最佳模型已导出 → {final_path}")
    else:
        print(f"\n⚠️  未找到最佳模型: {best_path}")

    print("\n✅ 训练完成!")


if __name__ == "__main__":
    main()
