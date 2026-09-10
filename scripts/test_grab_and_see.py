# scripts/test_grab_and_see.py
import math
import os
import sys
import time

import cv2
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.configs.config import CAM_EYE, CAM_HEIGHT, CAM_TARGET, CAM_UP, CAM_WIDTH, TABLE_SURFACE_Z
from src.control.ik_controller import move_to_pose
from src.env.sim_env import setup_simulation, init_panda_pose
from src.perception.camera import get_camera_image
from src.perception.detector import FakeDetector, visualize_result
from src.perception.pose_estimator import PoseEstimator

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Franka Panda 末端执行器配置
#  link 8  = panda_hand        ✅ 正确：真实物理 link，IK 可正常求解
#  link 11 = panda_grasptarget ❌ 错误：虚拟参考点，IK 奇异，肩关节不动
# ─────────────────────────────────────────────────────────────────────────────
END_EFFECTOR_INDEX = 8
# panda_hand 坐标系原点在手掌中心，到指尖约 0.105m
# IK 解到 hand 时，指尖还低 FINGER_OFFSET，所以目标 Z 需要补偿
FINGER_OFFSET = 0.105


def control_gripper(robot_id, open_width, steps=200, force=350):
    """控制夹爪开合。open_width: 0.0=全闭, 0.04=全开（单侧最大行程）"""
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


def capture_camera_image(filename):
    """
    渲染相机图像并保存。
    修复：img_arr[2] 在部分 PyBullet 版本下为 uint32（CV_32S），
    必须先 .astype(np.uint8) 再做 cvtColor，否则 OpenCV 报 Unsupported depth。
    """
    view_matrix = p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
    proj_matrix = p.computeProjectionMatrixFOV(60, CAM_WIDTH / CAM_HEIGHT, 0.1, 3.0)
    img_arr = p.getCameraImage(
        CAM_WIDTH,
        CAM_HEIGHT,
        view_matrix,
        proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    # ✅ 关键：先 astype(uint8)，再切片，再 cvtColor
    rgb = np.reshape(img_arr[2], (CAM_HEIGHT, CAM_WIDTH, 4)).astype(np.uint8)[:, :, :3]
    save_path = os.path.join(ASSETS_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"   📸 已保存快照 → {save_path}")
    return rgb


def get_ee_pos(robot_id):
    """获取当前末端执行器（panda_hand link8）的世界坐标"""
    state = p.getLinkState(robot_id, END_EFFECTOR_INDEX)
    return list(state[0])


def wait(steps=120):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    print("🚀 启动含 Panda 机械臂和视觉相机的环境...")
    client, robot_id, obj_id = setup_simulation(gui=True)

    # ── 初始化 Panda 预备姿态（必须，否则肩关节奇异不动） ──────────────────
    init_panda_pose(robot_id)

    # 等待物理引擎稳定（鸭子落稳 + 机械臂稳定）
    wait(steps=240)

    # 获取 GT 位置（仅用于对比）
    obj_pos_gt, _ = p.getBasePositionAndOrientation(obj_id)

    ee_pos = get_ee_pos(robot_id)
    print(
        f"📍 [GT] 鸭子位置: X={obj_pos_gt[0]:.3f}, Y={obj_pos_gt[1]:.3f}, Z={obj_pos_gt[2]:.3f}"
    )
    print(
        f"   初始末端 (link{END_EFFECTOR_INDEX}): "
        f"X={ee_pos[0]:.3f} Y={ee_pos[1]:.3f} Z={ee_pos[2]:.3f}"
    )
    print(f"   末端执行器: panda_hand (link={END_EFFECTOR_INDEX})")

    # ── 步骤 0：相机感知 ───────────────────────────────────────────────────
    print("\n👁️  步骤 0: 相机感知...")
    # 改用 get_camera_image() 获取 RGB + 深度 + 分割图
    rgb_bgr, depth_real, depth_vis, seg_img, view_matrix, proj_matrix = (
        get_camera_image()
    )

    # 注：capture_camera_image 已废弃，用 get_camera_image 替代
    # 但为了保持向后兼容，手动保存 RGB
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    save_path = os.path.join(ASSETS_DIR, "test_rgb.png")
    cv2.imwrite(save_path, rgb_bgr)
    save_path_depth = os.path.join(ASSETS_DIR, "test_depth.png")
    cv2.imwrite(save_path_depth, depth_vis)
    print(f"   📸 已保存快照 → {save_path}")
    print(f"   📏 已保存深度图 → {save_path_depth}")

    # ── 步骤 0.5：目标检测（FakeDetector） ───────────────────────────────
    print("🤖 步骤 0.5: 目标检测（FakeDetector）...")
    detector = FakeDetector()
    box, mask, cls_id = detector.detect(rgb)

    if box is not None:
        print(f"   🎯 检测成功 | 类别={cls_id} | BBox={box}")
        save_path = os.path.join(ASSETS_DIR, "detect_result.png")
        visualize_result(rgb_bgr, box, mask, save_path=save_path)
        print(f"   已保存: {save_path}")
    else:
        print("   ⚠️  未检测到物体，使用 GT 位置继续")
        cv2.imwrite(
            os.path.join(ASSETS_DIR, "detect_result.png"),
            rgb_bgr,
        )

    # ── 步骤 1：多视角采集 + 自标定 + 点云质心融合（无 GT）──────────────
    print("\n📐 步骤 1: 多视角点云质心融合（无 GT）...")

    duck_target = [obj_pos_gt[0], obj_pos_gt[1], obj_pos_gt[2]]
    # 融合视角：不同高度以打破 cy_offset 敏感度的对称性
    # 高度不同 → 图像 Y 轴映射到不同比例的世界 Y vs Z → LOO 可检测系统偏差
    fusion_views_config = [
        {"eye": [1.3,  0.0, 1.4], "target": duck_target},    # 视角 0：右侧，高位
        {"eye": [0.5, -1.0, 1.2], "target": duck_target},    # 视角 1：正前，中位
        {"eye": [-0.2,-0.5, 1.0], "target": duck_target},    # 视角 2：左前，低位
    ]

    # 几何分析视角：额外包含原相机视角，获取更完整的点云
    all_views_config = [
        {"eye": [1.0, -0.8, 1.3], "target": duck_target},    # 原相机：右前
    ] + fusion_views_config

    fusion_views = []
    for i, vc in enumerate(fusion_views_config):
        _, d_real, _, s_img, v_matrix, _ = get_camera_image(
            eye=vc["eye"], target=vc["target"]
        )
        fusion_views.append((d_real, s_img, v_matrix))
        print(f"   融合视角 {i}: eye={vc['eye']}")

    all_views = []
    for i, vc in enumerate(all_views_config):
        _, d_real, _, s_img, v_matrix, _ = get_camera_image(
            eye=vc["eye"], target=vc["target"]
        )
        all_views.append((d_real, s_img, v_matrix))
        if i == 0:
            print(f"   几何视角(原相机): eye={vc['eye']}")

    print(f"   融合 {len(fusion_views)} 视角（变高度）+ {len(all_views)} 视角几何分析")

    # 自标定：LOO 交叉验证 + 桌面约束锚定 Z 轴（无需 GT）
    print("   🔧 自标定中（LOO一致性 + 桌面约束）...")
    auto_calib = PoseEstimator.auto_calibrate_cy(
        fusion_views, obj_id, table_z=TABLE_SURFACE_Z
    )
    cy_offset = auto_calib["cy_offset"]
    print(
        f"   自标定: cy_offset={cy_offset:+d}, "
        f"LOO误差={auto_calib['loo_error_m'] * 100:.1f}cm"
    )

    estimator = PoseEstimator(cy_offset=cy_offset)
    result = estimator.estimate_multiview(fusion_views, obj_id)

    if result is not None:
        obj_pos = result["position_world"]
        pw, pg = obj_pos, np.array(obj_pos_gt[:3])
        err_cm = np.linalg.norm(pw - pg) * 100

        # 逐视角诊断
        print("\n   逐视角质心 (世界坐标):")
        pv = result.get("per_view_centroids", None)
        if pv is not None:
            for vi, c in enumerate(pv):
                e = np.linalg.norm(c - pg) * 100
                print(
                    f"     视角{vi}: X={c[0]:.4f} Y={c[1]:.4f} Z={c[2]:.4f}  误差={e:.1f}cm"
                )
            print(f"     → 加权中位数融合: X={pw[0]:.4f} Y={pw[1]:.4f} Z={pw[2]:.4f}")

        print(f"\n   融合质心: X={pw[0]:.4f}  Y={pw[1]:.4f}  Z={pw[2]:.4f}")
        print(f"   GT位置  : X={pg[0]:.4f}  Y={pg[1]:.4f}  Z={pg[2]:.4f}")
        print(
            f"   估计误差: {err_cm:.2f}cm  (融合 {result['total_points']} 点, "
            f"{result['views_used']}/{len(fusion_views)} 视角有效)"
        )
        # 不再回退 GT — 完全依赖估计
    else:
        print("❌ 估计失败，无法继续")
        p.disconnect()
        return

    # ── 抓取优化：多视角融合点云 → 几何分析 ──────────────────────────────
    print("\n🎯 抓取优化: 多视角融合点云几何分析...")

    # 使用所有视角（含原相机）合并点云 → 更完整的 3D 形状
    stats = estimator.get_multiview_point_cloud_stats(all_views, obj_id)

    if stats is not None:
        duck_z_min = stats["z_min"]
        duck_z_max = stats["z_max"]
        duck_z_range = stats["z_range"]
        long_axis = stats["long_axis"]

        print(
            f"   鸭子 Z 范围: {duck_z_min:.3f} ~ {duck_z_max:.3f} (高 {duck_z_range * 100:.1f}cm)"
        )
        print(f"   合并点云: {stats['num_points']} 点")
        print(f"   水平主轴: [{long_axis[0]:.3f}, {long_axis[1]:.3f}]")

        # Z 修正：质心偏向可视表面（俯视→顶部点数多→质心偏高），
        # 几何中点 (Z_min+Z_max)/2 对点密度不敏感，更接近真实体积中心
        z_mid = (duck_z_min + duck_z_max) / 2
        print(f"   几何中点 Z: {z_mid:.3f} (vs 质心 Z: {obj_pos[2]:.3f})")
        obj_pos[2] = z_mid
        err_corrected = np.linalg.norm(obj_pos - pg) * 100
        print(f"   Z修正后误差: {err_corrected:.1f}cm")

        # 抓取高度：身体最宽处 ≈ 15% 高度（从底部往上）
        grasp_ratio = 0.15
        grasp_z_body = duck_z_min + grasp_ratio * duck_z_range
        print(f"   推荐抓取 Z: {grasp_z_body:.3f} (身体 {grasp_ratio * 100:.0f}% 处)")

        # 夹爪偏航：垂直于物体水平主轴
        grasp_yaw = math.atan2(-long_axis[1], long_axis[0])
        print(f"   推荐夹爪偏航: {math.degrees(grasp_yaw):.0f}°")
    else:
        print("   ⚠️ 点云分析失败，使用质心默认值")
        duck_z_min = obj_pos[2]
        duck_z_max = obj_pos[2]
        duck_z_range = 0.1
        grasp_z_body = obj_pos[2]
        grasp_yaw = 0.0

    ox, oy = obj_pos[0], obj_pos[1]
    pre_grasp_z = grasp_z_body + FINGER_OFFSET + 0.20  # 预抓取：身体上方 20cm
    approach_z = grasp_z_body + FINGER_OFFSET + 0.05  # 接近点：身体上方 5cm
    fine_z = grasp_z_body + FINGER_OFFSET  # 指尖对准身体最宽处

    # 夹爪方向：指尖向下 + 自适应偏航（垂直物体主轴）
    down_quat = p.getQuaternionFromEuler([math.pi, 0, grasp_yaw])

    # ── 步骤 1.5：张开夹爪，移至预抓取点 ──────────────────────────────────
    print("\n🤖 步骤 1.5: 张开夹爪，移至目标正上方...")
    control_gripper(robot_id, open_width=0.04)

    pre_grasp_pos = [ox, oy, pre_grasp_z]
    print(f"   目标: X={ox:.3f} Y={oy:.3f} Z={pre_grasp_z:.3f}")
    move_to_pose(
        robot_id,
        pre_grasp_pos,
        target_quat=down_quat,
        end_effector_link_index=END_EFFECTOR_INDEX,
        steps=300,
    )
    ee = get_ee_pos(robot_id)
    err = math.dist(pre_grasp_pos, ee)
    print(
        f"   ✓ 预抓取点 | 实际({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | 误差={err * 100:.1f}cm"
    )

    # ── 步骤 2：下降至接近点 ─────────────────────────────────────────────
    print("\n⬇️  步骤 2: 下降至物体上方 50mm...")
    approach_pos = [ox, oy, approach_z]
    print(f"   目标: X={ox:.3f} Y={oy:.3f} Z={approach_z:.3f}")
    move_to_pose(
        robot_id,
        approach_pos,
        target_quat=down_quat,
        end_effector_link_index=END_EFFECTOR_INDEX,
        steps=300,
    )
    ee = get_ee_pos(robot_id)
    err = math.dist(approach_pos, ee)
    print(
        f"   ✓ 接近点 | 实际({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | 误差={err * 100:.1f}cm"
    )

    # ── 步骤 2.5：精细下降 ───────────────────────────────────────────────
    print("\n⏳ 步骤 2.5: 精细下降...")
    fine_pos = [ox, oy, fine_z]
    print(f"   目标: X={ox:.3f} Y={oy:.3f} Z={fine_z:.3f}")
    move_to_pose(
        robot_id,
        fine_pos,
        target_quat=down_quat,
        end_effector_link_index=END_EFFECTOR_INDEX,
        steps=200,
    )
    ee = get_ee_pos(robot_id)
    err = math.dist(fine_pos, ee)
    print(
        f"   ✓ 精细下降 | 实际({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) | 误差={err * 100:.1f}cm"
    )

    # 等待物理引擎稳定
    wait(steps=120)

    # ── 步骤 3：闭合夹爪 ────────────────────────────────────────────────
    print("\n✊ 步骤 3: 闭合夹爪...")
    control_gripper(robot_id, open_width=0.00, steps=200)

    # ── 步骤 3.5：夹紧稳定 ──────────────────────────────────────────────
    print("⏳ 步骤 3.5: 等待夹紧稳定...")
    wait(steps=120)

    # ── 步骤 3.6：轻提验证 ──────────────────────────────────────────────
    print("\n🔍 步骤 3.6: 轻提验证...")
    obj_z_before = p.getBasePositionAndOrientation(obj_id)[0][2]
    verify_pos = [ox, oy, fine_z + 0.05]
    move_to_pose(
        robot_id,
        verify_pos,
        target_quat=down_quat,
        end_effector_link_index=END_EFFECTOR_INDEX,
        steps=150,
    )
    wait(steps=120)

    obj_z_after = p.getBasePositionAndOrientation(obj_id)[0][2]
    delta_z = (obj_z_after - obj_z_before) * 100
    ee = get_ee_pos(robot_id)
    print(f"   ✓ 轻提 | 实际({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
    if delta_z > 1.0:
        print(f"   ✅ 抓取成功！物体上升 Δz={delta_z:.1f}cm")
    else:
        print(f"   ⚠️  物体未明显上升（Δz={delta_z:.1f}cm）")
        print(f"      → 可尝试将 FINGER_OFFSET 从 {FINGER_OFFSET} 调小 0.01~0.02")

    # ── 步骤 4：提升 ────────────────────────────────────────────────────
    print("\n⬆️  步骤 4: 提升物体...")
    lift_pos = [ox, oy, fine_z + 0.35]
    move_to_pose(
        robot_id,
        lift_pos,
        target_quat=down_quat,
        end_effector_link_index=END_EFFECTOR_INDEX,
        steps=400,
    )

    capture_camera_image("grasp_success.png")

    print("\n🎉 全流程完成！关闭窗口或 Ctrl+C 退出...")
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
