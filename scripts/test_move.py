# scripts/test_move.py
import os
import sys
import time

import pybullet as p

# 导入之前的环境和控制器
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.control.ik_controller import move_to_pose
from src.env.sim_env import setup_simulation


def main():
    print("🚀 启动仿真环境...")
    client, robot_id, obj_id = setup_simulation(gui=True)

    # 等待环境稳定
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 1. 获取方块位置
    print("📍 获取方块位置...")
    obj_pos, obj_quat = p.getBasePositionAndOrientation(obj_id)
    end_effector_index = 8  # panda_hand

    # 2. 移动到预抓取位置 (方块上方 15cm)
    print("🤖 步骤 1: 移动到方块上方...")
    pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
    move_to_pose(
        robot_id,
        pre_grasp_pos,
        target_quat=None,
        end_effector_link_index=end_effector_index,
        steps=200,
    )

    # 3. 垂直降落到方块上
    print("⬇️ 步骤 2: 降落接近方块...")
    # 稍微高出方块中心一点点，避免用力过猛把方块压飞
    grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02]
    move_to_pose(
        robot_id,
        grasp_pos,
        target_quat=None,
        end_effector_link_index=end_effector_index,
        steps=120,
    )

    # 4. 激活“吸盘” (创建固定约束)
    print("🧲 步骤 3: 激活电磁吸盘抓取！")
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_index,
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,  # -1 表示绑定到物体的 Base 上
        jointType=p.JOINT_FIXED,  # 固定连接
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    # 稍微等个零点几秒，让物理引擎把约束生效
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 5. 抬起方块
    print("⬆️ 步骤 4: 抓取成功，正在抬起方块...")
    lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.3]  # 抬高 30cm
    move_to_pose(
        robot_id,
        lift_pos,
        target_quat=None,
        end_effector_link_index=end_effector_index,
        steps=240,
    )

    print("🎉 Level 0 基础抓取闭环大功告成！")

    # 保持仿真运行让你欣赏杰作
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
