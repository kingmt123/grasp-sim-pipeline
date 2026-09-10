## 项目概述

PyBullet + Franka Panda 机械臂视觉引导抓取。使用 YOLOv8n-seg 替换 PyBullet seg_img 进行 mask 提取，
两级级联定位：默认相机粗估 → 4 视角迭代加权中位数融合 → 按"夹持面覆盖物体"物理规则推导的抓取高度。

> **STEP12（2026-09-10）**：修正内参模型与末端坐标系 4 个 bug（详见 `log.md` STEP12）。
> 本文档以下参数均已按修正后状态更新；若在旧提交/旧文档中看到 `cy_offset=-54`、
> `waist_ratio` 表、`ik_bias`、`FINGER_OFFSET=0.105`、`getLinkState()[0]`、
> "IK 工作空间极限 3.9cm"，那些均已废弃。

## 项目结构

```
src/
├── configs/config.py        # 所有参数配置（含 MULTI_OBJECTS_CONFIG）
├── control/ik_controller.py # 自适应 IK 位置控制（收敛判据用 link frame）
├── env/sim_env.py           # 仿真环境 (含 setup_simulation_multi)
├── perception/
│   ├── camera.py            # PyBullet 相机渲染 (RGB/深度/分割)
│   ├── detector.py          # 目标检测 (FakeDetector / YOLO, 含 detect_all)
│   └── pose_estimator.py    # 点云反投影、多视角融合、相机内参
└── pipeline/                # 管道封装
scripts/
├── test_grab_and_see.py         # 抓取流程（单物体 seg_img 版）
├── test_multi_object_grasp.py   # ⭐主抓取流程（多物体 YOLO 版）
├── test_multi_object.py         # 多物体环境加载验证
├── test_multi_object_detection.py # 多物体检测 + 位姿估计
├── generate_yolo_dataset.py     # YOLO 训练数据生成
├── train_yolo_custom.py         # 自定义 YOLO 训练
├── batch_test_final.py          # 批量统计：定位精度(vs GT AABB) + 抓取成功率 + --jitter
├── diagnose_z_bias.py           # 内参根因诊断（投影矩阵/重投影/rayTest/平面/逐视角一致性）
├── diagnose_finger_geometry.py  # 末端真实几何（CoM vs link frame、指尖、夹持面）
├── sweep_grasp_height.py        # 抓取高度扫描（重标用）
├── diagnose_duck_grasp.py       # 鸭子诊断
├── diagnose_centroid.py         # 单视角诊断（STEP10 前，已过时）
├── test_pose_estimation.py      # 位姿估计验证
└── test_move.py                 # 基础运动测试
models/
├── custom_yolov8n_seg.pt        # 训练好的 YOLO 模型
└── yolo_duck_teddy_cube/        # 训练日志
yolo_dataset/                    # 训练数据集
docs/                            # 文档（空，知识在 log.md / README.md / 本文件）
```

### 抓取高度（物理规则推导，不再 per-object 凑参数）

`_detect_waist_height()` 从合并点云中计算（实测手指夹持面高度 **0.062m**）：
- 物体比夹持面矮（cube 5cm）→ 指尖贴桌 `table_z + 5mm`，夹持面覆盖整个物体
- 物体比夹持面高（duck 8.5cm / teddy 9cm）→ 夹持面在物体上居中覆盖：`table_z + (h − 0.062)/2`
- `grasp_z` 即**真实指尖目标高度**；`fine_z = grasp_z + FINGER_OFFSET`（hand link frame 目标）
- `ik_bias` 仅保留 mm 级微调位（当前全为 0），cm 级残差由 fine 后的**闭环重发目标**处理
- 安全限制：`fine_z >= TABLE_SURFACE_Z + FINGER_OFFSET + 0.005`
  —— 内参 + link frame 修正后闭环残差 ≤2mm，该钳制因此作用在**实际到达位置**上
- 夹爪力度 500/800/1000N（per-object，避免 VHACD 碰撞导致 PyBullet 断开）

### 多物体 YOLO 定位
- 使用 YOLOv8n-seg 替代 PyBullet seg_img（无 GT 作弊）
- 两级级联: Stage1 默认相机粗估 XY → Stage2 4 视角 3D 加权融合（最多 **5** 次迭代，收敛判据 shift<2cm）
- **发散检测**: shift 连续 2 次增加时回退到最佳候选
- 加权融合: LOO 质量评分 × mask 大小作为权重 → 加权中位数（X/Y/Z 全 3D）
- `cy_offset = 0`（真主点即 240；旧值 −54 是内参模型错误的补偿，已随 STEP12 移除）
- 融合视角: 右侧[1.3,0,1.4]/正前[0.5,-1,1.2]/左侧[-0.2,-0.5,1.0]/左后[0.8,0.8,1.5]

### 相机内参（STEP12 修正，曾是 Z 偏高的根因）
- PyBullet `computeProjectionMatrixFOV` 的 `fov` 是**垂直** FOV：像素焦距
  `f = (CAM_HEIGHT/2)/tan(fov/2) = 415.69px`，主点 `(320, 240)`（投影矩阵 `m02=m12=0`）
- 图像行 v 向下增长而相机 +y 向上 → 反投影 `y_cam = -(v - cy) * depth / fy`（符号必须正确）
- 旧实现用 `(CAM_WIDTH/2)/tan` 使焦距偏大 **4/3 倍**，并靠 `cy_offset` 凑合 → Z 恒定偏高 8~12cm
- 校验工具：`scripts/diagnose_z_bias.py`（重投影自一致 + rayTest 命中率 95.7% / 3D 误差 0.18cm）

### 夹爪偏航（形状自适应）
- PCA 在多视角/单视角均不可行（透视畸造 / 点云圆形化）
- 改用物体类别硬编码先验：
  - teddy: **yaw=90°** — 躺桌上，手指沿 X 夹窄处
  - duck: **yaw=0°** — 手指沿 Y
  - cube: **yaw=0°** — 对称，无需旋转

### Panda 机械臂
- 末端执行器: link 8 (`panda_hand`), **不用** link 11 (`panda_grasptarget`)
- **末端位姿必须读 link frame**：`getLinkState(...)[4]`；`[0]` 是质心 CoM，比 link frame 低 3.90cm，
  读它会凭空造出"IK 残差 3.9cm"（STEP12 订正）
- **基座降低**: `ROBOT_BASE_POS = [0, 0, 0.5]`（原 0.625），给臂更多向下伸展空间
- `FINGER_OFFSET = 0.1162m`: IK 控制的 hand link frame → 真实指尖（手指碰撞 AABB 最低点，实测）；
  link frame → `panda_grasptarget` 是 0.1050m（另一个量，勿混用）
- 关节初始化: `[0, -0.3, 0, -2.0, 0, 1.8, 0.785]`
- 抓取分阶段: pre_grasp → approach → fine(600-800步) → 闭环Z微调(≤2轮) → grasp → verify → lift(可选)
- Per-object IK 参数:
  - duck: force=500N, fine_steps=600, threshold=2e-3
  - teddy: force=800N, fine_steps=600, threshold=2e-3
  - cube: force=1000N, fine_steps=800, threshold=1e-3

### 相机系统
- 虚拟相机: 640×480, FOV=60°, near=0.1m, far=3.0m
- OpenGL 坐标系: 相机 Z 朝屏幕外 (反投影时 z_cam = -depth)
- 渲染器：主线用 `ER_TINY_RENDERER`，数据集生成用 `ER_BULLET_HARDWARE_OPENGL`（不一致，待处理）

### 位置估计
- 两条数据流: 位置估计 (4 视角 3D 加权融合) + 抓取参数 (合并点云)
- 逐视角 3D 质心中位数（替代原始点云均值）；修正后逐视角 Z 中位互相一致到 0.3~0.4cm
- 抓取高度: 见上方物理规则
- 偏航: 物体类别硬编码（PCA 不可行）
- 桌面常数: `TABLE_SURFACE_Z = 0.625`（独立于 ROBOT_BASE_POS）
- 精度评分基准：XY 对 GT AABB 中心、Z 对 GT AABB 中点（可见表面质心的正确参照）

### 标定
- `auto_calibrate_cy()` 已过时：所谓"内参系统偏差"不存在，真主点即 (320, 240)，`cy_offset=0`
- 视角位姿原则: 同高度同距离 (1.0-1.4m), 避免俯视

## 运行命令

```bash
uv run python scripts/test_multi_object_grasp.py     # ⭐多物体 YOLO 抓取（主线）
uv run python scripts/test_grab_and_see.py            # 单物体 seg_img 版
uv run python scripts/test_cube_x50.py               # cube 独立测试（X=0.50，STEP10 前脚本）
uv run python scripts/test_multi_object.py            # 多物体环境加载验证
uv run python scripts/test_multi_object_detection.py  # 多物体检测+位姿估计
uv run python scripts/generate_yolo_dataset.py        # YOLO 训练数据生成
uv run python scripts/train_yolo_custom.py            # YOLO 训练
uv run python scripts/batch_test_final.py --pose-trials 5 --grasp-trials 3 --jitter 0.02  # 批量统计
uv run python scripts/diagnose_z_bias.py              # 内参根因诊断
uv run python scripts/diagnose_finger_geometry.py     # 末端几何实测
uv run python scripts/sweep_grasp_height.py --heights 0.60,0.62,0.64,0.66  # 抓取高度扫描
uv run python scripts/diagnose_duck_grasp.py          # 鸭子抓取诊断（STEP10 前脚本）
```
