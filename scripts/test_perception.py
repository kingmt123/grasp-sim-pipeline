import os
import sys

import matplotlib.image as mpimg

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.perception.detector import YoloSegmentor, visualize_result


def main():
    print("🔍 开始静态图片感知测试...")

    # 1. 读取之前保存的静态图片
    img_path = "assets/test_rgb.png"
    if not os.path.exists(img_path):
        print(f"❌ 找不到图片 {img_path}")
        print("   请先运行 test_grab_and_see.py 生成图片，或手动指定图片路径。")
        print("   尝试: python scripts/test_perception.py <your_image.png>")
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            img_path = sys.argv[1]
            print(f"   使用: {img_path}")
        else:
            return

    # matplotlib 读取的 png 通常是 0-1 的浮点数，转换为 0-255 的 uint8 格式
    img = mpimg.imread(img_path)
    rgb_image = (img[:, :, :3] * 255).astype("uint8")

    # 2. 实例化检测器（你可以切换 YoloSegmentor 来测试）
    # detector = FakeDetector()
    detector = YoloSegmentor(model_path="yolov8n-seg.pt")  # 第一次跑会下载权重

    # 3. 运行检测！
    box, mask, cls_id = detector.detect(rgb_image)

    print("🎯 检测成功！")
    print(f" - 类别 ID: {cls_id}")
    print(f" - BBox (左上x, 左上y, 右下x, 右下y): {box}")

    # 4. 可视化并保存
    visualize_result(rgb_image, box, mask, save_path="fake_detect_result.png")


if __name__ == "__main__":
    main()
