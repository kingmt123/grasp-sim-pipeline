# src/perception/camera.py
# 职责：PyBullet 相机渲染，提供 RGB/深度/分割图

import os
import sys

import cv2
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.configs.config import (
    CAM_EYE,
    CAM_FAR,
    CAM_FOV_DEG,
    CAM_HEIGHT,
    CAM_NEAR,
    CAM_TARGET,
    CAM_UP,
    CAM_WIDTH,
)


def depth_buffer_to_real(
    depth_buffer: np.ndarray, near: float, far: float
) -> np.ndarray:
    """
    OpenGL 归一化深度缓冲 [0,1] → 真实相机坐标系正轴深度（单位：米）。
    返回值是沿光轴方向的正距离（非 OpenGL 相机 -Z 方向的值）。
    """
    return far * near / (far - (far - near) * depth_buffer)


def get_camera_image(
    eye=None,
    target=None,
    up=None,
    fov=None,
    width=None,
    height=None,
    near=None,
    far=None,
    renderer=None,
):
    """
    Returns
    -------
    rgb_bgr      : (H, W, 3) uint8   BGR
    depth_real   : (H, W) float32    真实米制深度（正值，单位 m）
    depth_visual : (H, W) uint8      可视化用（近暗远亮）
    seg_img      : (H, W) int32      分割 ID 图
    view_matrix  : list[16]          列主序视图矩阵
    proj_matrix  : list[16]          列主序投影矩阵
    """
    eye = eye or CAM_EYE
    target = target or CAM_TARGET
    up = up or CAM_UP
    fov = fov or CAM_FOV_DEG
    width = width or CAM_WIDTH
    height = height or CAM_HEIGHT
    near = near or CAM_NEAR
    far = far or CAM_FAR
    renderer = renderer or p.ER_TINY_RENDERER

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far,
    )

    _, _, rgb_raw, depth_buf, seg_raw = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=renderer,
    )

    rgb_array = np.reshape(rgb_raw, (height, width, 4)).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)

    depth_buf_arr = np.reshape(depth_buf, (height, width)).astype(np.float32)
    depth_real = depth_buffer_to_real(depth_buf_arr, near, far)
    depth_visual = cv2.normalize(depth_real, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    seg_img = np.reshape(seg_raw, (height, width)).astype(np.int32)

    return rgb_bgr, depth_real, depth_visual, seg_img, view_matrix, proj_matrix
