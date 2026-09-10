# 自定义 YOLOv8n-seg 模型

训练好的 YOLO 分割权重放在这里。

## 当前模型

| 文件 | 说明 |
|------|------|
| `custom_yolov8n_seg.pt` | 自定义训练的 YOLOv8n-seg，识别 duck/teddy/cube |

## 训练指标

- Box mAP50: **0.995**
- Mask mAP50: **0.995**
- Box mAP50-95: **0.917**
- Mask mAP50-95: **0.881**
- 推理速度: ~42ms/张 (CPU Intel i7-14650HX)
- 参数量: 3.26M

## 类别映射

| class_id | 名称 | 训练实例数 |
|----------|------|-----------|
| 0 | duck 🦆 | 50 |
| 1 | teddy 🧸 | 49 |
| 2 | cube 🧊 | 50 |

## 训练方法

1. `scripts/generate_yolo_dataset.py` — 从 PyBullet 生成 200 张标注图
2. `scripts/train_yolo_custom.py` — 从 yolov8n-seg.pt 微调 80 epochs
3. 数据集在 `yolo_dataset/`，配置见 `yolo_dataset/dataset.yaml`
