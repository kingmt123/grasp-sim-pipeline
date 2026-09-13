# tests/test_regression.py
"""
回归测试：把 STEP12 修掉的 4 类缺陷固化成自动检查。

为什么这些测试值得存在
----------------------
这 4 个缺陷都属于"看起来对、跑得通、难发现"的类型，曾潜伏数月：

  #1 相机焦距用错 FOV 轴（PyBullet 的 fov 是垂直 FOV，应用 (H/2)/tan(fov/2)，
     代码用了 (W/2)/tan → 焦距偏大 4/3）
  #2 反投影 y 轴符号反了（图像行 v 向下增长，y_cam 必须取负）
  #3 末端位姿读的是 getLinkState()[0]（质心 CoM），而 IK 控制的是 link frame [4]
     —— 两者恒差 3.90cm，曾被误判为"IK 工作空间极限"
  #4 FINGER_OFFSET 与真实指尖几何不符（link frame → 手指 AABB 最低点 = 0.1162m）

任一被改回去，这里立刻红灯。全部测试用 p.DIRECT，无 GUI/显示依赖，可直接进 CI。
"""

import math
import os
import re

import numpy as np
import pytest

import pybullet as p

from src.configs.config import (
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_WIDTH,
    TABLE_SURFACE_Z,
)
from src.control.ik_controller import move_to_pose
from src.env.sim_env import init_panda_pose, setup_simulation_multi, stabilize_objects
from src.perception.pose_estimator import (
    PoseEstimator,
    get_camera_intrinsics,
    _pixels_to_world_points,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F_VERT = (CAM_HEIGHT / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)   # 415.69
F_LEGACY = (CAM_WIDTH / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)  # 554.26 (错误)
CAM_NEAR, CAM_FAR = 0.1, 3.0
EYE = [1.2, -0.8, 1.3]
TARGET = [0.5, 0.0, 0.65]
UP = [0.0, 0.0, 1.0]


def _V(m):
    return np.array(m, dtype=np.float64).reshape(4, 4).T


def _matrices():
    vm = p.computeViewMatrix(EYE, TARGET, UP)
    pm = p.computeProjectionMatrixFOV(CAM_FOV_DEG, CAM_WIDTH / CAM_HEIGHT, CAM_NEAR, CAM_FAR)
    return vm, pm


def _project_many(pts, vm, pm):
    M = _V(pm) @ _V(vm)
    h = np.column_stack([pts, np.ones(len(pts))])
    clip = (M @ h.T).T
    ndc = clip[:, :3] / clip[:, 3:4]
    return (ndc[:, 0] + 1) / 2 * CAM_WIDTH, (1 - ndc[:, 1]) / 2 * CAM_HEIGHT


@pytest.fixture
def sim():
    """DIRECT 模式的多物体场景（Panda + 桌子 + duck/teddy/cube），退出时断开。"""
    p.connect(p.DIRECT)
    _, robot_id, obj_infos = setup_simulation_multi(gui=False)
    init_panda_pose(robot_id)
    stabilize_objects(obj_infos, steps=40)
    yield robot_id, obj_infos
    p.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# #1 内参：必须与 PyBullet 自己的投影矩阵一致
# ─────────────────────────────────────────────────────────────────────────────
def test_intrinsics_match_pybullet_projection_matrix():
    p.connect(p.DIRECT)
    try:
        _, pm = _matrices()
        P = _V(pm)
        K = get_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV_DEG)
        assert K[0, 0] == pytest.approx(CAM_WIDTH / 2.0 * P[0, 0], abs=1e-3)
        assert K[1, 1] == pytest.approx(CAM_HEIGHT / 2.0 * P[1, 1], abs=1e-3)
        assert K[0, 2] == pytest.approx(CAM_WIDTH / 2.0 * (1 + P[0, 2]), abs=1e-3)
        assert K[1, 2] == pytest.approx(CAM_HEIGHT / 2.0 * (1 - P[1, 2]), abs=1e-3)
        # 垂直 FOV ⇒ 两个方向的像素焦距都等于 (H/2)/tan(fov/2)
        assert K[0, 0] == pytest.approx(F_VERT, abs=1e-3)
        assert K[1, 1] == pytest.approx(F_VERT, abs=1e-3)
        assert K[0, 2] == pytest.approx(CAM_WIDTH / 2.0, abs=1e-3)
        assert K[1, 2] == pytest.approx(CAM_HEIGHT / 2.0, abs=1e-3)
        # 不是旧模型（W/2 焦距）
        assert abs(K[0, 0] - F_LEGACY) > 100
    finally:
        p.disconnect()


def test_reprojection_self_consistency_with_correct_intrinsics():
    """正确内参下：像素 → 反投影 → 再投影必须回到同一像素（残差 < 1e-6 px）。"""
    p.connect(p.DIRECT)
    try:
        vm, pm = _matrices()
        rng = np.random.default_rng(0)
        xs = rng.uniform(0, CAM_WIDTH, 200)
        ys = rng.uniform(0, CAM_HEIGHT, 200)
        zs = rng.uniform(0.4, 2.5, 200)
        K = get_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV_DEG)
        pts = _pixels_to_world_points(xs, ys, zs, K[0, 0], K[1, 1], K[0, 2], K[1, 2], vm)
        u2, v2 = _project_many(pts, vm, pm)
        assert np.max(np.abs(u2 - xs)) < 1e-3
        assert np.max(np.abs(v2 - ys)) < 1e-3
    finally:
        p.disconnect()


def test_wrong_focal_length_or_y_sign_is_detected():
    """回归护栏：旧模型（f=W/2 与/或 y 取正）必须在本检查下失败。"""
    p.connect(p.DIRECT)
    try:
        vm, pm = _matrices()
        xs = np.array([400.0, 240.0, 100.0])
        ys = np.array([300.0, 360.0, 120.0])
        zs = np.array([1.0, 1.2, 0.8])

        def residual(fx, fy, sign):
            cam = np.column_stack([
                (xs - CAM_WIDTH / 2.0) * zs / fx,
                sign * (ys - CAM_HEIGHT / 2.0) * zs / fy,
                -zs,
            ])
            w = (np.linalg.inv(_V(vm)) @ np.column_stack([cam, np.ones(len(zs))]).T).T
            pts = w[:, :3] / w[:, 3:4]
            u2, v2 = _project_many(pts, vm, pm)
            return float(np.max(np.abs(np.r_[u2 - xs, v2 - ys])))

        # 正确：残差 ≈ 0
        assert residual(F_VERT, F_VERT, -1.0) < 1e-2
        # 焦距偏大 4/3：残差随离主轴距离线性增长（这里 > 20px）
        assert residual(F_LEGACY, F_LEGACY, -1.0) > 20.0
        # y 符号取正：镜像，残差 = 2|v - cy| 量级
        assert residual(F_VERT, F_VERT, +1.0) > 20.0
    finally:
        p.disconnect()


def test_unprojection_agrees_with_raycast_ground_truth(sim):
    """物理基准：mask 像素的相机射线命中点 vs 同像素深度反投影点（工具自带 GT）。"""
    robot_id, obj_infos = sim
    info = next(i for i in obj_infos if i["name"] == "teddy")
    aabb = p.getAABB(info["obj_id"])
    center = [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]
    vm, pm = _matrices()
    u, v = _project_many(np.array([center]), vm, pm)
    u, v = float(u[0]), float(v[0])
    if not (0 <= u < CAM_WIDTH and 0 <= v < CAM_HEIGHT):
        pytest.skip("目标不在视野内")

    K = get_camera_intrinsics(CAM_WIDTH, CAM_HEIGHT, CAM_FOV_DEG)
    # 该像素的相机射线（用被测模块自身的符号约定构造方向）
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0])
    R = _V(vm)[:3, :3]
    d = R.T @ (d_cam / np.linalg.norm(d_cam))
    eye = np.array(EYE, dtype=np.float64)
    hit = p.rayTest(eye.tolist(), (eye + d * 2.9).tolist())[0]
    assert hit[0] != -1, "射线未命中任何物体"
    p_true = np.array(hit[3])

    _, _, rgb, depth_buf, _ = p.getCameraImage(
        width=CAM_WIDTH, height=CAM_HEIGHT, viewMatrix=vm, projectionMatrix=pm)
    depth = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) *
                                  np.reshape(depth_buf, (CAM_HEIGHT, CAM_WIDTH)))
    p_est = _pixels_to_world_points(
        np.array([u]), np.array([v]), np.array([depth[int(v), int(u)]]),
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], vm)[0]
    assert np.linalg.norm(p_est - p_true) < 0.02, (
        f"深度反投影与 rayTest 真值偏差 {np.linalg.norm(p_est - p_true) * 100:.2f}cm "
        f"(内参 y 符号/焦距一旦回退，这里会立刻失败)"
    )


def test_pose_estimator_defaults_have_no_cy_fudge_and_vertical_fov():
    """回归护栏：cy_offset 必须为 0（真主点 240），焦距必须是垂直 FOV 值。"""
    est = PoseEstimator(verbose=False)
    assert est.cy_offset == 0
    assert est.cy == pytest.approx(CAM_HEIGHT / 2.0)
    assert est.fx == pytest.approx(F_VERT, abs=1e-3)
    assert est.fy == pytest.approx(F_VERT, abs=1e-3)
    # 主点不在中心是不可能的：旧代码依赖 240-54=186
    assert est.cy != pytest.approx(CAM_HEIGHT / 2.0 - 54)


# ─────────────────────────────────────────────────────────────────────────────
# #3 末端坐标系：IK 控制 link frame，读 CoM 会凭空多出 ~3.9cm
# ─────────────────────────────────────────────────────────────────────────────
def test_ik_converges_on_link_frame_not_center_of_mass(sim):
    robot_id, _ = sim
    target = [0.50, 0.0, 0.85]
    down = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
    move_to_pose(robot_id, target, target_quat=down, end_effector_link_index=8,
                 steps=400, convergence_threshold=2e-3)
    st = p.getLinkState(robot_id, 8)
    link_frame = np.array(st[4])
    com = np.array(st[0])
    assert np.linalg.norm(link_frame - np.array(target)) < 0.01, "IK 未收敛到 link frame"
    assert np.linalg.norm(com - np.array(target)) > 0.02, (
        "CoM 与 link frame 应当相差 ~3.9cm —— 若此断言失败说明坐标系语义变了，"
        "请重新确认 get_ee_pos / ik_controller 用的是 [4]"
    )


def test_finger_offset_matches_measured_geometry(sim, grasp_main):
    """回归护栏：FINGER_OFFSET 必须等于实测的 link frame → 手指 AABB 最低点距离。"""
    robot_id, _ = sim
    down = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
    move_to_pose(robot_id, [0.50, 0.0, 0.80], target_quat=down,
                 end_effector_link_index=8, steps=400, convergence_threshold=2e-3)
    hand_z = p.getLinkState(robot_id, 8)[4][2]
    tips = []
    for idx in (9, 10):
        aabb = p.getAABB(robot_id, linkIndex=idx)
        tips.append(aabb[0][2])
    measured = hand_z - min(tips)
    assert grasp_main.FINGER_OFFSET == pytest.approx(measured, abs=0.003), (
        f"FINGER_OFFSET={grasp_main.FINGER_OFFSET} 与实测 {measured:.4f}m 不符"
    )
    assert grasp_main.FINGER_OFFSET != pytest.approx(0.105, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP12 的抓取高度规则（夹持面覆盖物体）+ 类别映射一致性
# ─────────────────────────────────────────────────────────────────────────────
def test_grasp_height_rule_uses_pad_coverage(grasp_main):
    rng = np.random.default_rng(1)

    def cloud(top_z):
        z = np.concatenate([rng.uniform(TABLE_SURFACE_Z, top_z, 900), np.full(100, top_z)])
        xy = rng.uniform(-0.02, 0.02, (len(z), 2))
        return np.column_stack([xy, z])

    # 物体比夹持面(0.062m)矮 → 指尖贴桌（桌面 + 5mm），夹持面覆盖整个物体
    assert grasp_main._detect_waist_height(
        cloud(TABLE_SURFACE_Z + 0.050), 0.0, TABLE_SURFACE_Z,
        grasp_main.FINGER_OFFSET) == pytest.approx(TABLE_SURFACE_Z + 0.005, abs=0.003)
    # 物体更高 → 夹持面在物体上居中覆盖
    expect = TABLE_SURFACE_Z + (0.090 - 0.062) / 2.0
    assert grasp_main._detect_waist_height(
        cloud(TABLE_SURFACE_Z + 0.090), 0.0, TABLE_SURFACE_Z,
        grasp_main.FINGER_OFFSET) == pytest.approx(expect, abs=0.003)


def test_class_id_mapping_matches_dataset_yaml(grasp_main):
    """YOLO 类别顺序与训练集 dataset.yaml 必须一致（静默不一致会抓错物体）。"""
    path = os.path.join(REPO_ROOT, "yolo_dataset", "dataset.yaml")
    text = open(path, encoding="utf-8").read()
    names = {int(m.group(1)): m.group(2).strip()
             for m in re.finditer(r"^\s*(\d+)\s*:\s*(\S+)", text, flags=re.M)}
    assert names, "dataset.yaml 未解析出类别名"
    assert {v: k for k, v in names.items()} == grasp_main.YOLO_CLASS_IDS


# ─────────────────────────────────────────────────────────────────────────────
# 端到端感知 smoke（用真实 YOLO 权重；权重不在仓库时跳过）
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_estimate_object_pose_accuracy_smoke(grasp_main):
    model = os.path.join(REPO_ROOT, "models", "custom_yolov8n_seg.pt")
    if not os.path.exists(model):
        pytest.skip("YOLO 权重不在仓库中")
    from src.perception.detector import YoloSegmentor

    p.connect(p.DIRECT)
    try:
        _, robot_id, obj_infos = setup_simulation_multi(gui=False)
        init_panda_pose(robot_id)
        stabilize_objects(obj_infos, steps=40)
        det = YoloSegmentor(model_path=model)
        for name in ("duck", "teddy", "cube"):
            info = next(i for i in obj_infos if i["name"] == name)
            aabb = p.getAABB(info["obj_id"])
            gt_xy = np.array([(aabb[0][0] + aabb[1][0]) / 2, (aabb[0][1] + aabb[1][1]) / 2])
            gt_z_mid = (aabb[0][2] + aabb[1][2]) / 2
            est, _ = grasp_main.estimate_object_pose(
                name, grasp_main.YOLO_CLASS_IDS[name], [0.45, 0.0, 0.65], det)
            assert est is not None, f"{name} 位姿估计失败"
            xy_err = float(np.linalg.norm(np.array(est[:2]) - gt_xy))
            z_err = float(est[2] - gt_z_mid)
            assert xy_err < 0.04, f"{name} XY 误差 {xy_err * 100:.1f}cm 超限"
            assert abs(z_err) < 0.03, f"{name} Z 偏差 {z_err * 100:+.1f}cm 超限（内参回归？）"
    finally:
        p.disconnect()
