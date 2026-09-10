# src/env/sim_env.py
import math
import os
import sys
import time

import pybullet as p
import pybullet_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.configs.config import (
    MULTI_OBJECTS_CONFIG,
    OBJ_INIT_POS,
    ROBOT_BASE_POS,
    ROBOT_BASE_QUAT,
    TABLE_POS,
)


def setup_simulation(gui=True):
    """单物体模式：Panda + 桌子 + 小黄鸭。"""
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=TABLE_POS)

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=ROBOT_BASE_POS,
        baseOrientation=ROBOT_BASE_QUAT,
        useFixedBase=True,
    )

    # 使用 load_single_object 加载鸭子（从 config 读角度，单位统一为度）
    obj_id = load_single_object(
        "duck_vhacd.urdf",
        OBJ_INIT_POS,
        [90, 0, 0],     # 绕 X 轴 +90° 站立，单位度
        1.5,
    )

    print(f"   [sim_env] robot_id={robot_id}, obj_id={obj_id}")
    return client, robot_id, obj_id


def init_panda_pose(robot_id):
    """
    初始化 Panda 到合理预备姿态。
    不初始化直接从零位运动，肩关节处于奇异点，IK 无法求解大幅运动。
    """
    init_angles = [0, -0.3, 0, -2.0, 0, 1.8, 0.785]
    for i, angle in enumerate(init_angles):
        p.resetJointState(robot_id, i, angle)
    p.resetJointState(robot_id, 9, 0.04)  # 左指张开
    p.resetJointState(robot_id, 10, 0.04)  # 右指张开
    for _ in range(50):
        p.stepSimulation()


def stabilize_objects(obj_infos, steps=240):
    """
    多物体稳定：reset 到位姿 → 降低弹性 → 物理步进 → 强制归位。
    解决 VHACD 物体同时加载时互相弹飞的问题。
    """
    for info in obj_infos:
        p.resetBasePositionAndOrientation(info["obj_id"], info["pos"], [0, 0, 0, 1])
        p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])
        # VHACD 物体需要低弹性 + 高摩擦防止弹跳和漂移
        friction = 1.5 if info["name"] == "duck" else 0.8
        p.changeDynamics(info["obj_id"], -1, restitution=0.0, lateralFriction=friction)

    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 物理步进后只锁定 XY，让 Z 自然落桌（避免强制归位引发弹跳）
    for info in obj_infos:
        pos, quat = p.getBasePositionAndOrientation(info["obj_id"])
        p.resetBasePositionAndOrientation(
            info["obj_id"],
            [info["pos"][0], info["pos"][1], pos[2]],  # 锁定 XY，保留自然 Z
            quat,
        )
        p.resetBaseVelocity(info["obj_id"], [0, 0, 0], [0, 0, 0])


def load_single_object(urdf_name, pos, euler_deg, scaling):
    """
    加载单个物体。

    Parameters
    ----------
    urdf_name : str
    pos : list[float] 3
    euler_deg : list[float] 3  欧拉角，单位**度**
    scaling : float
    """
    quat = p.getQuaternionFromEuler([math.radians(a) for a in euler_deg])
    obj_id = p.loadURDF(
        urdf_name,
        basePosition=pos,
        baseOrientation=quat,
        globalScaling=scaling,
    )
    return obj_id


def setup_simulation_multi(gui=True, objects_config=None):
    """
    多物体仿真环境。

    Parameters
    ----------
    objects_config : list[dict], optional
        每个 dict 含 name/urdf/pos/euler/scaling。
        默认使用 config.MULTI_OBJECTS_CONFIG。

    Returns
    -------
    client, robot_id, obj_infos : list[dict]
        每个 dict 含 name/obj_id/pos/urdf
    """
    if objects_config is None:
        objects_config = MULTI_OBJECTS_CONFIG

    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=TABLE_POS)

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=ROBOT_BASE_POS,
        baseOrientation=ROBOT_BASE_QUAT,
        useFixedBase=True,
    )

    obj_infos = []
    for cfg in objects_config:
        obj_id = load_single_object(cfg["urdf"], cfg["pos"], cfg["euler"], cfg["scaling"])
        obj_infos.append({
            "name": cfg["name"],
            "obj_id": obj_id,
            "pos": cfg["pos"],
            "urdf": cfg["urdf"],
        })
        print(f"   [sim_env] 加载物体 [{cfg['name']}]  → obj_id={obj_id}  @ {cfg['pos']}")

    print(f"   [sim_env] robot_id={robot_id}, 共 {len(obj_infos)} 个物体")
    return client, robot_id, obj_infos


if __name__ == "__main__":
    print("用法: python -m src.env.sim_env [single|multi]")
    mode = sys.argv[1] if len(sys.argv) > 1 else "multi"

    if mode == "multi":
        print("多物体模式")
        client, robot_id, obj_infos = setup_simulation_multi(gui=True)
        print(f"\n=== 物体列表 ===")
        for info in obj_infos:
            print(f"  [{info['name']}] obj_id={info['obj_id']} @ {info['pos']}")
    else:
        print("单物体模式")
        client, robot_id, obj_id = setup_simulation(gui=True)

        print("\n=== Panda Link 列表 ===")
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            print(f"  Link {i:2d}: {info[12].decode():35s}  joint={info[1].decode()}")

    print("\n按 Ctrl+C 退出")
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
