## 项目概述

PyBullet + Franka Panda 机械臂视觉引导抓取。使用 YOLOv8n-seg 替换 PyBullet seg_img 进行 mask 提取，
三级级联定位：默认相机粗估 → 3 视角多指向中位数融合 → 固定桌面高度抓取。

## 项目结构

```
src/
├── configs/config.py        # 所有参数配置（含 MULTI_OBJECTS_CONFIG）
├── control/ik_controller.py # 自适应 IK 位置控制
├── env/sim_env.py           # 仿真环境 (含 setup_simulation_multi)
├── perception/
│   ├── camera.py            # PyBullet 相机渲染 (RGB/深度/分割)
│   ├── detector.py          # 目标检测 (FakeDetector / YOLO, 含 detect_all)
│   └── pose_estimator.py    # 点云反投影、多视角融合、相机标定
└── pipeline/                # 管道封装
scripts/
├── test_grab_and_see.py         # 主抓取流程（单物体 seg_img 版）
├── test_multi_object_grasp.py   # ⭐主抓取流程（多物体 YOLO 版）
├── test_multi_object.py         # 多物体环境加载验证
├── test_multi_object_detection.py # 多物体检测 + 位姿估计
├── generate_yolo_dataset.py     # YOLO 训练数据生成
├── train_yolo_custom.py         # 自定义 YOLO 训练
├── batch_test_final.py          # 批量测试
├── diagnose_duck_grasp.py       # 鸭子诊断
├── diagnose_centroid.py         # 单视角诊断
├── test_pose_estimation.py      # 位姿估计验证
└── test_move.py                 # 基础运动测试
models/
├── custom_yolov8n_seg.pt        # 训练好的 YOLO 模型
└── yolo_duck_teddy_cube/        # 训练日志
yolo_dataset/                    # 训练数据集
docs/                            # 文档
```

### 抓取高度（物体形状自适应）

`_detect_waist_height()` 从合并点云中计算：
- `grasp_z = table_z + waist_ratio × (z_max - table_z)`（以桌面为绝对基准）
- 使用 `table_z` 而非 YOLO 点云 `z_min`：YOLO mask 腐蚀让最低点偏高
- Per-object waist_ratio:
  - teddy: **25%** — 夹身体偏下，避尾巴尖
  - duck: **20%** — 略高于底部，夹腹部
  - cube: **15%** — 矮小对称，底部稳定
- `fine_z = grasp_z + FINGER_OFFSET + ik_bias`（hand 目标）
- 安全限制：`fine_z >= TABLE_SURFACE_Z + FINGER_OFFSET + 0.005`（指尖不低于桌面 5mm）
- 夹爪力度 350-1000N（per-object，避免 VHACD 碰撞导致 PyBullet 断开）

### 多物体 YOLO 定位
- 使用 YOLOv8n-seg 替代 PyBullet seg_img（无 GT 作弊）
- 四级迭代: Stage1 默认相机粗估 → Stage2 4 视角 3D 加权融合（最多 **5** 次迭代）
- **发散检测**: shift 连续 2 次增加时回退到最佳候选，防止 Y 偏差发散
- 加权融合: LOO 质量评分 × mask 大小作为权重 → 加权中位数（X/Y/Z 全 3D）
- cy_offset=-54 适用于所有物体
- 融合视角: 右侧[1.3,0,1.4]/正前[0.5,-1,1.2]/左侧[-0.2,-0.5,1.0]/左后[0.8,0.8,1.5]

### 夹爪偏航（形状自适应）
- PCA 在多视角/单视角均不可行（透视畸造 / 点云圆形化）
- 改用物体类别硬编码先验：
  - teddy: **yaw=90°** — 躺桌上，手指沿 X 夹窄处
  - duck: **yaw=0°** — 手指沿 Y
  - cube: **yaw=0°** — 对称，无需旋转

### Panda 机械臂
- 末端执行器: link 8 (`panda_hand`), **不用** link 11 (`panda_grasptarget`)
- **基座降低**: `ROBOT_BASE_POS = [0, 0, 0.5]`（原 0.625），给臂更多向下伸展空间
- `FINGER_OFFSET = 0.105m`: IK 解到 hand 位置，指尖在 hand 下方 0.105m
- 关节初始化: `[0, -0.3, 0, -2.0, 0, 1.8, 0.785]`
- 抓取分阶段: pre_grasp → approach → fine(600-800步) → grasp → verify → lift(可选)
- Per-object IK 参数:
  - duck: force=500N, fine_steps=600, threshold=2e-3, ik_bias=-0.020
  - teddy: force=800N, fine_steps=600, threshold=2e-3, ik_bias=0.000
  - cube: force=1000N, fine_steps=800, threshold=1e-3, ik_bias=-0.020

### 相机系统
- 虚拟相机: 640×480, FOV=60°, near=0.1m, far=3.0m
- OpenGL 坐标系: 相机 Z 朝屏幕外 (反投影时 z_cam = -depth)
- `cy_offset` 不能跨不同朝向的视角共用

### 位置估计
- 两条数据流: 位置估计 (4 视角 3D 加权融合) + 抓取参数 (合并点云)
- GT-Free 自标定: LOO-XY + 桌面约束 (TABLE_SURFACE_Z=0.625)
- 逐视角 3D 质心中位数（替代原始点云均值）
- 抓取高度: `table_z + waist_ratio × (z_max - table_z)`（以桌面为绝对基准）
- 偏航: 物体类别硬编码（PCA 不可行）
- 桌面常数: `TABLE_SURFACE_Z = 0.625`（独立于 ROBOT_BASE_POS）

### 标定
- `auto_calibrate_cy()`: 无需 GT 的自标定
- 视角位姿原则: 同高度同距离 (1.0-1.4m), 避免俯视

## 运行命令

```bash
uv run python scripts/test_multi_object_grasp.py     # ⭐多物体 YOLO 抓取（主线）
uv run python scripts/test_grab_and_see.py            # 单物体 seg_img 版
uv run python scripts/test_cube_x50.py               # cube 独立测试（X=0.50）
uv run python scripts/test_multi_object.py            # 多物体环境加载验证
uv run python scripts/test_multi_object_detection.py  # 多物体检测+位姿估计
uv run python scripts/generate_yolo_dataset.py        # YOLO 训练数据生成
uv run python scripts/train_yolo_custom.py            # YOLO 训练
uv run python scripts/batch_test_final.py             # 批量性能测试
uv run python scripts/diagnose_duck_grasp.py          # 鸭子抓取诊断
```
