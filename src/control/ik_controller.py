# src/control/ik_controller.py
import time

import numpy as np
import pybullet as p


def move_to_pose(
    robot_id,
    target_pos,
    target_quat=None,
    end_effector_link_index=8,
    steps=300,
    adaptive_force=True,
    convergence_threshold=5e-3,
    verbose=False,
):
    """
    Panda IK 控制器。

    rest_poses 设计原则（针对基座在 X=0、目标在 X≈0.5 的正前方场景）：
    ┌────────┬────────────────────────────────────────────────────────────┐
    │ 关节   │ 含义与取值理由                                             │
    ├────────┼────────────────────────────────────────────────────────────┤
    │ j1=0.0 │ 基座旋转：0 = 朝正 +X，目标就在正前方，不需要转腰          │
    │ j2=-0.3│ 肩部：轻微前倾（-0.3 ≈ -17°），适合 0.5m 距离前方目标    │
    │        │ 原值 -0.785(-45°) 让肩部过度后仰，末端偏后 ~30cm          │
    │ j3=0.0 │ 上臂旋转：保持中立                                         │
    │ j4=-1.5│ 肘部：-1.5 ≈ -86°，肘部自然弯曲朝下，末端能到桌面高度    │
    │        │ 原值 -2.356(-135°) 肘部过度弯曲，末端上抬                  │
    │ j5=0.0 │ 前臂旋转：中立                                             │
    │ j6=1.8 │ 腕部俯仰：1.8 ≈ 103°，配合 down_quat 让夹爪垂直向下      │
    │        │ 原值 1.571(90°) 略显不足，1.8 更接近垂直向下自然姿态      │
    │ j7=0.0 │ 腕部旋转：0 = 对称，不偏左右                              │
    └────────┴────────────────────────────────────────────────────────────┘
    关节 8-11（夹爪及虚拟关节）补 0，必须补全否则 PyBullet null-space 优化失效。
    """
    movable_joints = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    max_velocities = []

    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] != p.JOINT_FIXED:
            movable_joints.append(i)
            lower_limits.append(info[8])
            upper_limits.append(info[9])
            joint_ranges.append(max(info[9] - info[8], 0.01))
            max_velocities.append(max(info[11], 0.1))

    # 7 个臂关节的 rest pose（正前方 0.5m 目标专用）
    arm_rest = [0.0, -0.3, 0.0, -1.5, 0.0, 1.8, 0.0]
    # 补全至 movable_joints 总数（夹爪和虚拟关节填 0）
    rest_poses = arm_rest + [0.0] * max(0, len(movable_joints) - 7)

    last_ik_poses = None
    converge_count = 0
    CONVERGE_NEED = 8  # 连续 8 步误差 < 阈值才算收敛

    for step in range(steps):
        # ── IK 求解 ────────────────────────────────────────────────
        ik_kwargs = dict(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=end_effector_link_index,
            targetPosition=target_pos,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=300,
            residualThreshold=1e-5,
        )
        if target_quat is not None:
            ik_kwargs["targetOrientation"] = target_quat

        ik_poses = p.calculateInverseKinematics(**ik_kwargs)

        if not ik_poses:
            if last_ik_poses is None:
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
                continue
            ik_poses = last_ik_poses
        else:
            last_ik_poses = ik_poses

        # ── 当前末端位置与误差 ─────────────────────────────────────
        ee_pos = np.array(p.getLinkState(robot_id, end_effector_link_index)[0])
        err = np.linalg.norm(ee_pos - np.array(target_pos))

        # ── 自适应力和速度 ─────────────────────────────────────────
        if err < 0.03:
            force, vel = 200, 0.3
        elif err < 0.08:
            force, vel = 350, 0.8
        else:
            force, vel = 500, 1.5

        # ── 控制前 7 个臂关节 ──────────────────────────────────────
        for idx, jidx in enumerate(movable_joints[:7]):
            angle = float(ik_poses[idx])
            angle = max(lower_limits[idx], min(upper_limits[idx], angle))
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=jidx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=force,
                maxVelocity=min(vel, max_velocities[idx] * 0.6),
            )

        # ── 收敛检测 ───────────────────────────────────────────────
        if err < convergence_threshold:
            converge_count += 1
            if converge_count >= CONVERGE_NEED:
                if verbose:
                    print(f"  ✅ 收敛 step={step}, 误差={err * 100:.1f}cm")
                break
        else:
            converge_count = 0

        if verbose and step % 60 == 0:
            print(f"  step={step:3d} | err={err * 100:.1f}cm | F={force}N | V={vel}")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)
