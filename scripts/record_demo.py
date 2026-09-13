# scripts/record_demo.py
"""
录制抓取演示（无屏幕录制、无 GUI 窗口、可重复生成）。

做法
----
不改动主线代码：运行时打三个"演示专用"补丁
  1. p.connect → 强制 DIRECT（headless，无需桌面）
  2. p.getCameraImage → 强制 ER_TINY_RENDERER（DIRECT 下没有 GL 上下文）
  3. p.stepSimulation → 每 N 步抓一帧（相机固定俯视视角，叠加当前进度文字）
然后照常调用主线的 main()，产出 MP4（高质量）+ GIF（README 用，体积受控）。

用法
----
    uv run python scripts/record_demo.py
    uv run python scripts/record_demo.py --out docs/demo.gif --mp4 docs/demo.mp4 --every 8
"""

import argparse
import builtins
import importlib.util
import math
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import pybullet as p  # noqa: E402

CAM_W, CAM_H = 640, 480
VIEW_EYE = [1.15, -0.80, 1.22]
VIEW_TARGET = [0.52, 0.0, 0.70]
VIEW_UP = [0.0, 0.0, 1.0]
TITLE = "PyBullet + Panda | YOLOv8n-seg + 4-view fusion"


def load_main():
    path = os.path.join(REPO_ROOT, "scripts", "test_multi_object_grasp.py")
    spec = importlib.util.spec_from_file_location("grasp_main_demo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grasp_main_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "demo.gif"))
    ap.add_argument("--mp4", default=os.path.join(REPO_ROOT, "docs", "demo.mp4"))
    ap.add_argument("--every", type=int, default=8, help="每 N 个仿真步抓一帧")
    ap.add_argument("--gif-stride", type=int, default=5, help="（PIL 回退用）抽帧步长")
    ap.add_argument("--gif-width", type=int, default=440)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-speed", type=float, default=0.5,
                    help="播放速度系数（0.5 = 2 倍速；仿真节奏慢，加速后更适合 README）")
    ap.add_argument("--gif-colors", type=int, default=128)
    args = ap.parse_args()

    frames, state = [], {"step": 0, "label": "loading scene...", "ok": 0}

    _connect, _get_cam, _step = p.connect, p.getCameraImage, p.stepSimulation

    def _capture():
        vm = p.computeViewMatrix(VIEW_EYE, VIEW_TARGET, VIEW_UP)
        pm = p.computeProjectionMatrixFOV(60.0, CAM_W / CAM_H, 0.1, 3.0)
        _, _, rgb, _, _ = _get_cam(CAM_W, CAM_H, vm, pm, renderer=p.ER_TINY_RENDERER)
        img = np.reshape(rgb, (CAM_H, CAM_W, 4)).astype(np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 顶栏：标题 + 成功计数（ASCII 标签：cv2 HERSHEY 字体无法渲染中文）
        cv2.rectangle(img, (0, 0), (CAM_W, 32), (28, 28, 28), -1)
        cv2.putText(img, TITLE, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(img, f'grasped {state["ok"]}/3', (CAM_W - 108, CAM_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 235, 120), 1, cv2.LINE_AA)
        # 底栏：当前阶段
        cv2.rectangle(img, (0, CAM_H - 30), (CAM_W, CAM_H), (28, 28, 28), -1)
        cv2.putText(img, state["label"], (10, CAM_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _connect_direct(*a, **kw):
        # 主线内部按 gui=True 连接 GUI；演示一律强制 DIRECT（headless）
        return _connect(p.DIRECT)

    def _get_cam_sw(*a, **kw):
        kw["renderer"] = p.ER_TINY_RENDERER
        return _get_cam(*a, **kw)

    def _step_spy(*a, **kw):
        r = _step(*a, **kw)
        state["step"] += 1
        if state["step"] % args.every == 0:
            frames.append(_capture())
        return r

    # ---- 从主线的打印里解析当前阶段，叠加到画面上（仅演示用）----
    obj_re = re.compile(r"物体 \d+/\d+: (\w+)")
    _print = builtins.print
    last_obj = {"name": ""}

    def _print_spy(*a, **kw):
        text = " ".join(str(x) for x in a)
        m = obj_re.search(text)
        if m:
            last_obj["name"] = m.group(1)
        if "Stage1:" in text:
            state["label"] = f"[{last_obj['name']}] YOLO segmentation + point cloud"
        elif "收敛" in text or "最终 (" in text:
            state["label"] = f"[{last_obj['name']}] 4-view fusion localisation converged"
        elif "开始抓取" in text:
            state["label"] = f"[{last_obj['name']}] IK approach + fine descent"
        elif "手指接触" in text:
            state["label"] = f"[{last_obj['name']}] gripper closing (contact detected)"
        elif "抓取成功" in text:
            state["ok"] += 1
            state["label"] = f"[{last_obj['name']}] lift verified - SUCCESS"
        elif "未明显上升" in text:
            state["label"] = f"[{last_obj['name']}] not lifted - retry"
        return _print(*a, **kw)

    p.connect, p.getCameraImage, p.stepSimulation = _connect_direct, _get_cam_sw, _step_spy
    builtins.print = _print_spy
    MAIN = load_main()
    try:
        MAIN.main()
    finally:
        p.connect, p.getCameraImage, p.stepSimulation = _connect, _get_cam, _step
        builtins.print = _print

    if not frames:
        raise SystemExit("未抓到任何帧：请检查 --every 与仿真是否真的运行")

    print(f"\n🎬 抓到 {len(frames)} 帧（每 {args.every} 步一帧）")

    # ---- MP4（全帧）----
    os.makedirs(os.path.dirname(args.mp4), exist_ok=True)
    vw = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (CAM_W, CAM_H))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"💾 MP4 → {args.mp4} ({os.path.getsize(args.mp4) / 1e6:.2f} MB)")

    # ---- GIF（优先 ffmpeg palettegen；回退 PIL）----
    # PIL 量化效率差（4~8MB）；ffmpeg 双 pass 调色板通常 <2MB，且可同时加速播放。
    gif_ok = False
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        vf = (f"setpts={args.gif_speed}*PTS,fps={args.gif_fps},"
              f"scale={args.gif_width}:-1:flags=lanczos,split[s0][s1];"
              f"[s0]palettegen=max_colors={args.gif_colors}[p];"
              f"[s1][p]paletteuse=dither=bayer")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.mp4,
                            "-vf", vf, "-loop", "0", args.out], capture_output=True, text=True)
        gif_ok = r.returncode == 0 and os.path.exists(args.out)
        if not gif_ok:
            print(f"⚠️ ffmpeg 转 GIF 失败，回退 PIL：{r.stderr.strip()[:200]}")
    if not gif_ok:
        from PIL import Image
        sel = frames[::args.gif_stride]
        if len(sel) > 200:
            sel = sel[:200]
        scale = args.gif_width / CAM_W
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize(
            (args.gif_width, int(CAM_H * scale)), Image.LANCZOS) for f in sel]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.gif_fps), loop=0, optimize=True,
                     colors=args.gif_colors)
    print(f"💾 GIF → {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
