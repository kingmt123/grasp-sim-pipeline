import cv2  # 如果没安装，需在终端运行 pip install opencv-python
import numpy as np
import torch
from ultralytics import YOLO


class FakeDetector:
    """
    假检测器 (Dummy Detector)
    不依赖任何深度学习模型，直接在画面中央偏下位置硬编码生成一个检测框。
    用于打通视觉与控制的 Pipeline 接口。
    """

    def detect(self, rgb_image):
        h, w, _ = rgb_image.shape

        # 假设方块大概在画面的这个比例位置 (你可以根据实际照片微调这些参数)
        x_min, x_max = int(w * 0.45), int(w * 0.55)
        y_min, y_max = int(h * 0.60), int(h * 0.80)

        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # 生成掩码 (Mask)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 1

        cls_id = 0  # 0 代表我们想抓的方块

        return box, mask, cls_id


def visualize_result(image, box, mask, save_path="detect_result.png"):
    """
    可视化工具：将检测框画在图片上并保存。
    mask 参数预留用于后续叠加分割掩码可视化。
    """
    # PyBullet 传出的是 RGB，OpenCV 保存需要 BGR，做一次转换
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 提取坐标并画绿色边界框
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 在框的上方写上标签
    cv2.putText(
        img_bgr,
        "Target_Box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    cv2.imwrite(save_path, img_bgr)
    print(f"✅ 视觉检测结果已渲染并保存至: {save_path}")


# ===================================================================


class YoloSegmentor:
    """
    真实的 AI 视觉感知器 (基于 YOLOv8)
    加载 YOLO 模型进行目标检测和实例分割。

    训练自定义模型:
        1. uv run python scripts/generate_yolo_dataset.py   # 生成仿真数据集
        2. uv run python scripts/train_yolo_custom.py        # 微调模型
        3. 加载训练好的权重: model_path="models/custom_yolov8n_seg.pt"

    预训练 COCO 模型只认得 80 类常见物体（人、车、杯子等），
    不认得仿真小黄鸭，建议用自己的训练模型。
    """

    def __init__(self, model_path="yolov8n-seg.pt", device=None):
        # 自动检测是否有显卡 (GPU)，没有就用 CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⏳ 正在加载 YOLOv8 模型 ({model_path}) 到 {self.device}...")

        # 第一次运行会自动从 GitHub 下载 yolov8n-seg.pt 权重文件 (约 6MB)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print("✅ YOLOv8 模型加载完毕！")

    def detect(self, rgb_image):
        # 运行推理 (YOLO 原生支持 numpy 的 RGB 图片)
        results = self.model(
            rgb_image, verbose=False, conf=0.1
        )[
            0
        ]  # 由于图中只有一个物体，我们直接取第一个结果 [0]，并设置较低的置信度阈值 (conf=0.1) 来确保检测到目标。

        # 如果没检测到任何东西
        if len(results.boxes) == 0:
            print("⚠️ YOLO 未检测到任何物体！")
            return None, None, None

        # 提取第一个被检测到的物体的 Bounding Box [x_min, y_min, x_max, y_max]
        box = results.boxes.xyxy[0].cpu().numpy()

        # 提取类别 ID (例如人是 0，杯子是 41，书是 73)
        cls_id = int(results.boxes.cls[0].cpu().numpy())

        # 提取掩码 Mask (如果是分割模型)
        mask = None
        if results.masks is not None:
            mask = results.masks.data[0].cpu().numpy()

        return box, mask, cls_id

    def detect_all(self, rgb_image):
        """
        检测图像中所有物体（多物体支持）。

        Parameters
        ----------
        rgb_image : np.ndarray (H, W, 3) uint8 RGB

        Returns
        -------
        list[dict] : 每个元素含 box/mask/cls_id/conf/class_name
            若没有检测到任何物体，返回空列表。
        """
        results = self.model(rgb_image, verbose=False, conf=0.1)[0]

        if len(results.boxes) == 0:
            print("⚠️ YOLO 未检测到任何物体！")
            return []

        detections = []
        for i in range(len(results.boxes)):
            box = results.boxes.xyxy[i].cpu().numpy()
            cls_id = int(results.boxes.cls[i].cpu().numpy())
            conf = float(results.boxes.conf[i].cpu().numpy())
            class_name = results.names[cls_id] if hasattr(results, 'names') else str(cls_id)

            mask = None
            if results.masks is not None and i < len(results.masks.data):
                mask = results.masks.data[i].cpu().numpy()

            detections.append({
                "box": box,
                "mask": mask,
                "cls_id": cls_id,
                "conf": conf,
                "class_name": class_name,
            })

        return detections
