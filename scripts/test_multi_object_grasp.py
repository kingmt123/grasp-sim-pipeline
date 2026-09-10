# scripts/test_multi_object_grasp.py
"""
第三步：多视角融合 + 多物体抓取（YOLO 驱动）。
对每个物体：3 视角采集 → YOLO 分割提取 mask → 多视角融合位姿
→ 点云几何分析 → pre_grasp → approach → fine → grasp → verify → lift

关键特性：
  - 物体掩码：使用自定义 YOLOv8n-seg 模型（models/custom_yolov8n_seg.pt），
    完全替代 PyBullet seg_img，不依赖任何 GT 坐标。
  - 相机指向：使用固定桌面中心 [0.45, 0, 0.65]，不依赖物体实际位置。
  - IK 目标：完全来自 YOLO + 多视角融合估计。
  - GT 坐标（getBasePosition）仅用于验证轻提，不参与抓取决策。
"""

import math
import os
import sys
import time

import cv2
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import (
    CAM_FOV_DEG, CAM_HEIGHT, CAM_WIDTH, CAM_NEAR, CAM_FAR,
    ROBOT_BASE_POS, TABLE_SURFACE_Z,
)
from src.control.ik_controller import move_to_pose
from src.env.sim_env import (
    setup_simulation_multi,
    init_panda_pose,
    stabilize_objects,
)
from src.perception.camera import get_camera_image
from src.perception.pose_estimator import PoseEstimator
from src.perception.detector import YoloSegmentor

# ── YOLO 类别映射 ───────────────────────────────────────────────
# 与 yolo_dataset/dataset.yaml 一致: 0=duck, 1=teddy, 2=cube
YOLO_CLASS_IDS = {"duck": 0, "teddy": 1, "cube": 2}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Panda 末端执行器配置 ─────────────────────────────────────────────
END_EFFECTOR_INDEX = 8   # panda_hand
FINGER_OFFSET = 0.105     # hand 关节原点到指尖

# 融合视角（变高度，指向目标物体）
FUSION_VIEWS_CONFIG = [
    {"eye": [1.3,  0.0, 1.4]},   # 右侧高位
    {"eye": [0.5, -1.0, 1.2]},   # 正前中位
    {"eye": [-0.2,-0.5, 1.0]},   # 左前低位
]

# ── 辅助函数 ─────────────────────────────────────────────────────────

def get_ee_pos(robot_id):
    state = p.getLinkState(robot_id, END_EFFECTOR_INDEX)
    return list(state[0])


def control_gripper(robot_id, open_width, steps=200, force=350):
    """控制夹爪开合。open_width: 0.0=全闭, 0.04=全开。"""
    for joint_index in [9, 10]:
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=open_width,
            force=force,
        )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def wait(steps=120):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def estimate_object_pose(obj_name, target_class_id, _obj_pos_rough, yolo_detector):
    """
    两级级联 YOLO + 多视角中位数融合定位。

    Stage 1: 默认相机 YOLO → 粗略 XY (用于多视角指向)
    Stage 2: 多视角指向 Stage1 XY → YOLO mask → 点云
             → 逐视角3D质心 → 加权中位数融合
    """
    FIXED_CY = -54
    estimator = PoseEstimator(cy_offset=FIXED_CY)

    def _mask_to_pts(rgb_bgr, depth, view_matrix):
        """YOLO mask → 点云 → (pts, px_count) or (None, 0)。"""
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        dets = yolo_detector.detect_all(rgb)
        target = next((d for d in dets if d["cls_id"] == target_class_id), None)
        if target is None:
            return None, 0
        mask = cv2.resize(target["mask"], (CAM_WIDTH, CAM_HEIGHT),
                          interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 0.5).astype(np.uint8)
        mask_bin = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
        px = np.sum(mask_bin)
        if px < 100:
            return None, px
        fake_seg = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.int32)
        fake_seg[mask_bin > 0] = 9999
        pts = estimator.extract_view_points(depth, fake_seg, view_matrix, 9999)
        if pts is None or len(pts) < 30:
            return None, px
        return pts, px

    # ── Stage 1: 默认相机粗估 XY ─────────────────────────────────
    rgb1, depth1, _, _, vm1, _ = get_camera_image()
    pts1, px1 = _mask_to_pts(rgb1, depth1, vm1)
    if pts1 is None:
        print(f"   ❌ [{obj_name}] Stage1 YOLO 未检测到")
        return None, None

    rough_xy = np.median(pts1[:, :2], axis=0)
    print(f"   [{obj_name}] Stage1: {px1}px → XY=({rough_xy[0]:.3f}, {rough_xy[1]:.3f})")

    # ── Stage 2: 多视角融合（迭代至收敛） ──────────────────────────
    # 最多 5 次迭代 + 发散检测：Stage1 在 Y 方向偏差较大（~9cm）
    # 时需更多迭代收敛；若 shift 连续 2 次增加则恢复最佳候选并终止
    FUSION_EYES = [
        [1.3,  0.0, 1.4],   # 右侧高位
        [0.5, -1.0, 1.2],   # 正前中位
        [-0.2,-0.5, 1.0],   # 左前低位
        [0.8,  0.8, 1.5],   # 左后高位
    ]

    best_xy = rough_xy.copy()
    best_pts = pts1
    # pre-init new_pos, guard fused_z NameError
    new_pos = np.array([float(best_xy[0]), float(best_xy[1]), float(np.mean(best_pts[:, 2]))])
    # divergence guard: track best shift, restore if worsen 2x consecutively
    best_shift = float('inf')
    best_candidate_xy = best_xy.copy()
    best_candidate_pts = best_pts
    best_candidate_new_pos = new_pos.copy()
    diverging_count = 0
    for iteration in range(5):  # 3->5 iter for Y-biased Stage1
        cam_target = [float(best_xy[0]), float(best_xy[1]), 0.65]
        centroids_3d = []      # 存储每个视角 3D 中位数 (融合 X+Y+Z)
        pts_list_local = []
        mask_sizes = []

        for vi, eye in enumerate(FUSION_EYES):
            rgb, depth, _, _, vm, _ = get_camera_image(eye=eye, target=cam_target)
            pts, px = _mask_to_pts(rgb, depth, vm)
            if pts is not None and len(pts) >= 30:
                centroids_3d.append(np.median(pts, axis=0))  # 3D 中位数
                pts_list_local.append(pts)
                mask_sizes.append(px)

        if len(centroids_3d) < 2:
            print(f"     iter{iteration}: 有效视角不足 ({len(centroids_3d)})，停止迭代")
            if iteration == 0:
                # Stage2 无任何有效视角 → 回退 Stage1 结果
                print(f"     回退至 Stage1 粗估结果")
            break

        # LOO + mask-size 加权中位数（全 3D 融合）
        c3 = np.array(centroids_3d)
        ms = np.array(mask_sizes, dtype=np.float64)
        n = len(c3)

        if n >= 3:
            loo = np.ones(n, dtype=np.float64)
            for i in range(n):
                others = np.delete(c3, i, axis=0)
                dist = max(np.linalg.norm(c3[i] - np.median(others, axis=0)), 1e-6)
                loo[i] = 1.0 / dist
            loo /= loo.sum()
            weights = loo * ms
        else:
            weights = ms
        weights /= weights.sum()

        new_pos = np.zeros(3)
        for dim in range(3):
            order = np.argsort(c3[:, dim])
            cum_w = np.cumsum(weights[order])
            idx = min(np.searchsorted(cum_w, 0.5), n - 1)
            new_pos[dim] = c3[order[idx], dim]

        shift = np.linalg.norm(new_pos[:2] - best_xy) * 100
        w_str = np.round(weights * 100).astype(int)
        loo_str = np.round(loo * 100).astype(int) if n >= 3 else [0] * n
        print(f"     iter{iteration}: {n}视角 权重{w_str}% LOO{loo_str}% "
              f"→ ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}) shift={shift:.1f}cm")

        best_xy = new_pos[:2]
        best_pts = np.vstack(pts_list_local) if len(pts_list_local) > 1 else pts_list_local[0]

        # track best convergence, detect divergence
        if shift < best_shift:
            best_shift = shift
            best_candidate_xy = best_xy.copy()
            best_candidate_pts = best_pts
            best_candidate_new_pos = new_pos.copy()
            diverging_count = 0
        else:
            diverging_count += 1
            if diverging_count >= 2 and iteration >= 2:
                print(f"       发散 (shift 连续{diverging_count}次增加)，回退至 iter{iteration-diverging_count} 结果")
                best_xy = best_candidate_xy.copy()
                best_pts = best_candidate_pts
                new_pos = best_candidate_new_pos.copy()
                break

        # 收敛判断：shift < 2cm 停止迭代
        if shift < 2.0 and iteration > 0:
            print(f"       收敛 (shift={shift:.1f}cm < 2cm)")
            break

    final_pts = best_pts
    fused_xy = best_xy
    fused_z = new_pos[2]  # ✅ 来自逐视角 3D 中位数加权融合，而非原始点云均值

    # ── 最终位置 ──────────────────────────────────────────────────
    position_world = np.array([fused_xy[0], fused_xy[1], fused_z])

    # ── 物体形状自适应抓取高度 ────────────────────────────────────
    grasp_z = _detect_waist_height(final_pts, 0.0, TABLE_SURFACE_Z, FINGER_OFFSET, obj_name=obj_name)
    print(f"   [{obj_name}] 抓取高度: Z={grasp_z:.3f}m")

    # ── 物体形状自适应夹爪偏航 ──────────────────────────────────
    # 点云 PCA 不可靠：
    #   - 单视角：透视投影制造虚假长轴方向，cube 被误判为偏航≠0
    #   - 多视角融合：点云被圆形化，各方向方差接近
    # 改用物体类别硬编码先验（用户确认 teddy=90° 夹窄处最合理）
    #
    #    teddy (躺桌上): yaw=90° → 手指沿 X 开合，夹身体窄处
    #    duck  (异形):   yaw=0°  → 手指沿 Y 开合
    #    cube  (正方体): yaw=0°  → 对称，无需旋转
    grasp_yaw = {
        "teddy": math.pi / 2,
        "duck": 0.0,
        "cube": 0.0,
    }.get(obj_name, 0.0)

    print(f"   [{obj_name}] 最终 ({position_world[0]:.3f}, {position_world[1]:.3f}, "
          f"{position_world[2]:.3f}) 偏航={math.degrees(grasp_yaw):.0f}° "
          f"(形状自适应, {len(final_pts)}点)")

    return position_world, {
        "grasp_z": grasp_z,       # 物体形状自适应 Z（不含 finger_offset）
        "grasp_yaw": grasp_yaw,
        "z_min": TABLE_SURFACE_Z,
        "z_range": float(np.max(final_pts[:, 2]) - TABLE_SURFACE_Z),
    }


def _detect_waist_height(pts, grasp_yaw, table_z, finger_offset, obj_name=None, bin_size=0.005):
    """
    物体形状自适应抓取高度。

    不同物体形状差异大，统一用底部 15% 会导致：
      - teddy（高~12cm）：指尖只夹到尾巴尖
      - duck（异形）：偏低碰桌
      - cube（矮~4cm）：15% 合理

    改用物体特定 waist_ratio，使指尖夹在合理位置。

    Parameters
    ----------
    obj_name : str or None  物体名，决定 waist_ratio

    Returns
    -------
    grasp_z : float     抓取高度 Z（手指目标，不含 finger_offset）
    """
    if len(pts) < 50:
        return float(np.mean(pts[:, 2]))

    # 使用桌面高度作为绝对基准（YOLO mask 腐蚀可能让点云最低点高于真实底部）
    z_max = float(np.percentile(pts[:, 2], 95))
    z_range = max(z_max - table_z, 0.005)

    if z_range < 0.02:
        return table_z + 0.005

    # ── 物体形状自适应比例 ─────────────────────────────────────
    # teddy: 25% — 夹身体偏下，避尾巴尖
    # duck:  20% — 略高于底部，夹腹部
    # cube:  15% — 矮小对称，底部稳定
    WAIST_RATIOS = {
        "teddy": 0.25,
        "duck":  0.20,
        "cube":  0.15,
    }
    waist_ratio = WAIST_RATIOS.get(obj_name)
    if waist_ratio is None:
        # fallback: z_range 自适应
        if z_range > 0.08:
            waist_ratio = 0.30
        elif z_range < 0.04:
            waist_ratio = 0.15
        else:
            waist_ratio = 0.15 + 0.15 * (z_range - 0.04) / 0.04

    grasp_z = table_z + waist_ratio * z_range
    return grasp_z  # 手指目标 Z，不含 finger_offset

def grasp_object(robot_id, obj_id, obj_name, pos_world, grasp_params):
    """
    对单个物体执行完整抓取流程。

    关键改进：
    - grasp_params["grasp_z"] 现在返回腰线 Z（手指目标，不含 FINGER_OFFSET）
    - fine_z = waist_z + FINGER_OFFSET  手爪目标
    - 增加 IK 系统偏差补偿（Panda 在低 Z 区有 ~3.9cm 残余误差）

    Returns
    -------
    bool : 是否成功
    """
    ox, oy = pos_world[0], pos_world[1]
    grasp_z = grasp_params["grasp_z"]      # 物体形状自适应 Z（手指目标位置）
    grasp_yaw = grasp_params["grasp_yaw"]

    # 夹爪方向：指尖向下 + 自适应偏航
    down_quat = p.getQuaternionFromEuler([math.pi, 0, grasp_yaw])

    # ── 抓取参数（分物体，来自诊断脚本 proven 值）───────────────────
    # ik_bias: IK 在 hand 目标 Z > 0.76 时有 ~3.9cm 残差（工作空间边缘）。
    #   duck: 负 bias → 目标偏低 → IK 残差~3.9cm → 指尖靠近桌面
    #   teddy: 零 bias → 目标偏高 → IK 残差~3.9cm → 指尖在物体中部 (更合理的夹取方式)
    #   cube: 负 bias → VHACD 侧面接触 (小物体)
    obj_params = {
        "duck":  {"force": 500, "fine_steps": 600, "threshold": 2e-3, "appr_gap": 0.05, "ik_bias": -0.020},
        "teddy": {"force": 800, "fine_steps": 600, "threshold": 2e-3, "appr_gap": 0.05, "ik_bias": 0.000},
        "cube":  {"force": 1000, "fine_steps": 800, "threshold": 1e-3, "appr_gap": 0.03, "ik_bias": -0.020},
    }
    pset = obj_params.get(obj_name, obj_params["duck"])

    # ── 下降高度计算 ───────────────────────────────────────────────
    fine_z = grasp_z + FINGER_OFFSET + pset["ik_bias"]

    # ── 安全限制：指尖不低于桌面 5mm ────────────────────────────
    min_fine_z = TABLE_SURFACE_Z + FINGER_OFFSET + 0.005
    if fine_z < min_fine_z:
        print(f"      ⚠️ fine_z 过底 ({fine_z:.3f})，抬升至 {min_fine_z:.3f}")
        fine_z = min_fine_z

    appr_z = fine_z + pset["appr_gap"]
    pre_z = fine_z + 0.20

    print(f"\n   🤖 [{obj_name}] 开始抓取:")
    print(f"      抓取高 Z={grasp_z:.3f}, hand目标={fine_z:.3f}, "
          f"指尖预计={grasp_z:.3f}")
    print(f"      pre_grasp @ ({ox:.3f}, {oy:.3f}, {pre_z:.3f})")

    # 1. 张开夹爪
    control_gripper(robot_id, open_width=0.04)

    # 2. 预抓取
    move_to_pose(robot_id, [ox, oy, pre_z], target_quat=down_quat,
                 end_effector_link_index=END_EFFECTOR_INDEX, steps=300)
    ee = get_ee_pos(robot_id)
    err = math.dist([ox, oy, pre_z], ee)
    print(f"      ✓ pre_grasp | err={err*100:.1f}cm")

    # 3. 接近
    print(f"      approach  @ ({ox:.3f}, {oy:.3f}, {appr_z:.3f})")
    move_to_pose(robot_id, [ox, oy, appr_z], target_quat=down_quat,
                 end_effector_link_index=END_EFFECTOR_INDEX, steps=300)
    ee = get_ee_pos(robot_id)
    err = math.dist([ox, oy, appr_z], ee)
    print(f"      ✓ approach | err={err*100:.1f}cm")

    # 4. 精细下降（分物体参数）
    print(f"      fine      @ ({ox:.3f}, {oy:.3f}, {fine_z:.3f})  steps={pset['fine_steps']} thresh={pset['threshold']}")
    move_to_pose(robot_id, [ox, oy, fine_z], target_quat=down_quat,
                 end_effector_link_index=END_EFFECTOR_INDEX,
                 steps=pset["fine_steps"], convergence_threshold=pset["threshold"])
    ee = get_ee_pos(robot_id)
    err = math.dist([ox, oy, fine_z], ee)
    print(f"      ✓ fine     | err={err*100:.1f}cm  hand_Z={ee[2]:.3f}  fingertip_Z={ee[2]-FINGER_OFFSET:.3f}")

    wait(steps=60)

    # 5. 增加接触摩擦（物体 + 夹爪手指）
    p.changeDynamics(obj_id, -1, lateralFriction=2.0, spinningFriction=1.0)
    for j in [9, 10]:
        p.changeDynamics(robot_id, j, lateralFriction=5.0, spinningFriction=2.0)

    # 6. 闭合夹爪（分物体力度，来自诊断脚本 proven 值）
    grasp_force = pset["force"]
    print(f"      ✊ grasp (force={grasp_force}N)")
    control_gripper(robot_id, open_width=0.00, steps=300, force=grasp_force)
    # 额外等待让手指充分贴合物体表面
    for _ in range(150):
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL,
                                targetPosition=0.0, force=grasp_force)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL,
                                targetPosition=0.0, force=grasp_force)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    wait(steps=120)

    # ---- 微调阶段：检测手指接触，无接触则逐次下降重抓 ----
    MAX_RETRIES = 10
    DESCENT_STEP = 0.002  # 每次下降 2mm
    contact_made = False

    for retry in range(MAX_RETRIES + 1):
        # 检测手指 (link 9,10) 与物体之间的接触
        contacts = p.getContactPoints(robot_id, obj_id)
        finger_contacts = [c for c in contacts if c[3] in (9, 10)]

        if finger_contacts:
            contact_made = True
            print(f"      👆 手指接触 (retry {retry}): "
                  f"{len(finger_contacts)} contacts, dist={min(c[8] for c in finger_contacts):+.3f}")
            break

        if retry == MAX_RETRIES:
            print(f"      ⚠️  {MAX_RETRIES}次微调后仍无手指接触")
            break

        # 无接触 → 张开，微调下降，重新闭合
        if retry == 0:
            current_z = ee[2]
        else:
            current_z -= DESCENT_STEP

        new_z = current_z - DESCENT_STEP
        # 安全检查：finger 不能贯穿桌面
        if new_z - FINGER_OFFSET < TABLE_SURFACE_Z + 0.002:
            print(f"      ⚠️  finger 已达桌面高度，停止微调")
            break

        # 微调：张开，下降到新位置，合拢
        control_gripper(robot_id, open_width=0.04, steps=80, force=50)
        move_to_pose(robot_id, [ox, oy, new_z], target_quat=down_quat,
                     end_effector_link_index=END_EFFECTOR_INDEX, steps=100,
                     convergence_threshold=1e-2)
        control_gripper(robot_id, open_width=0.00, steps=200, force=grasp_force)
        # 保持指力
        for _ in range(60):
            p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=grasp_force)
            p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=grasp_force)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        if retry % 3 == 2:
            print(f"      ↻ retry {retry+1}/{MAX_RETRIES}: hand_Z={new_z:.3f} "
                  f"fingertip_Z={new_z-FINGER_OFFSET:.3f}")

    # ---- 轻提验证 ----
    obj_z_before = p.getBasePositionAndOrientation(obj_id)[0][2]
    verify_z = fine_z + 0.05
    move_to_pose(robot_id, [ox, oy, verify_z], target_quat=down_quat,
                 end_effector_link_index=END_EFFECTOR_INDEX, steps=150)
    wait(steps=120)

    obj_z_after = p.getBasePositionAndOrientation(obj_id)[0][2]
    delta_z = (obj_z_after - obj_z_before) * 100

    if delta_z > 1.0:
        print(f"      ✅ 抓取成功！物体上升 Δz={delta_z:.1f}cm")

        # 先拍照 — 即使后续崩溃也有记录
        save_path = os.path.join(ASSETS_DIR, f"grasp_{obj_name}.png")
        view_matrix = p.computeViewMatrix(
            [1.2, -0.8, 1.3], [ox, oy, verify_z], [0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            CAM_FOV_DEG, CAM_WIDTH / CAM_HEIGHT, CAM_NEAR, CAM_FAR
        )
        img_arr = p.getCameraImage(
            CAM_WIDTH, CAM_HEIGHT, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(img_arr[2], (CAM_HEIGHT, CAM_WIDTH, 4)).astype(np.uint8)[:, :, :3]
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"      📸 快照 → {save_path}")

        # 平滑提升 + 松开（避免突然的夹爪张开和大幅关节摆动）
        # 先继续抬高 15cm，给物体足够的离地空间
        lift_z = verify_z + 0.15
        move_to_pose(robot_id, [ox, oy, lift_z], target_quat=down_quat,
                     end_effector_link_index=END_EFFECTOR_INDEX, steps=300)
        # 在较高位置松开夹爪（物体下落距离短，视觉更自然）
        control_gripper(robot_id, open_width=0.04, steps=100)
        wait(steps=60)

        return True
    else:
        print(f"      ⚠️  物体未明显上升（Δz={delta_z:.1f}cm），松开重试")
        control_gripper(robot_id, open_width=0.04, steps=100)
        return False


def main():
    print("=" * 60)
    print("  🚀 第三步：多视角融合 + 多物体抓取")
    print("=" * 60)

    # ── 1. 加载环境 + YOLO ──────────────────────────────────────────
    print("\n📦 加载多物体环境...")
    client, robot_id, obj_infos = setup_simulation_multi(gui=True)
    init_panda_pose(robot_id)
    stabilize_objects(obj_infos, steps=240)
    print(f"   已加载 {len(obj_infos)} 个物体")

    # 加载 YOLO 自定义模型（用于 mask 提取，替代 PyBullet seg_img）
    print("\n🤖 加载 YOLO 自定义模型...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "models", "custom_yolov8n_seg.pt")
    yolo_detector = YoloSegmentor(model_path=model_path)
    print("   ✅ YOLO 就绪")

    # ── 2. 逐物体抓取 ────────────────────────────────────────────────
    results = []
    for idx, info in enumerate(obj_infos):
        obj_name = info["name"]
        obj_id = info["obj_id"]  # 仅用于 verify，不参与位姿估计
        # 固定相机指向桌面中心
        obj_pos_rough = [0.45, 0.0, 0.65]

        print(f"\n{'=' * 50}")
        print(f"  🎯 物体 {idx+1}/{len(obj_infos)}: {obj_name}")
        print(f"{'=' * 50}")

        # 回到预备姿态（每次抓取后）
        if idx > 0:
            init_panda_pose(robot_id)
            wait(steps=60)

        # YOLO 驱动的多视角位姿估计
        pos_world, grasp_params = estimate_object_pose(
            obj_name, YOLO_CLASS_IDS[obj_name], obj_pos_rough, yolo_detector
        )
        if pos_world is None:
            print(f"   ❌ [{obj_name}] 位姿估计失败，跳过")
            results.append({"name": obj_name, "success": False, "error": "pose_estimation_failed"})
            continue

        print(f"\n   📍 [{obj_name}] 估计: ({pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f})")

        # 执行抓取
        try:
            success = grasp_object(robot_id, obj_id, obj_name, pos_world, grasp_params)
        except Exception as e:
            print(f"   ❌ [{obj_name}] 抓取异常: {e}")
            success = False
            # 如果 PyBullet 断开，停止后续物体
            if "Not connected" in str(e):
                print("   ⚠️  PyBullet 断开，终止抓取")
                results.append({"name": obj_name, "success": False, "error": "simulation_disconnected"})
                break
        results.append({"name": obj_name, "success": success})

        # 如果 PyBullet 断开，停止
        try:
            p.getNumJoints(robot_id)
        except:
            print("   ⚠️  PyBullet 断开，终止")
            break

    # ── 3. 汇总 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  📊 多物体抓取完成")
    print(f"{'=' * 60}")
    success_count = sum(1 for r in results if r["success"])
    print(f"\n  成功: {success_count}/{len(results)}")
    for r in results:
        status = "✅ 成功" if r["success"] else "❌ 失败"
        print(f"    [{r['name']}] {status}")

    # 保持查看
    for _ in range(480):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
    print("\n👋 退出")


if __name__ == "__main__":
    main()
