#!/usr/bin/env python3
"""把某个 X11 窗口录成 mp4（本机没装 ffmpeg，用 Pillow 截屏 + OpenCV 写视频）。

等名为 --name 的窗口出现 → 按它在屏幕上的位置每秒录 --fps 帧（每 2 秒重查一次位置，窗口被拖动也跟着录）→
窗口关掉超过 3 秒就停、保存。录的是屏幕那块区域：被别的窗口挡住会把挡住的东西也录进去。

    python record_window.py --name "RobotDecision closed loop" --out ~/atom_episodes/videos/xxx.mp4
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


def geometry(name):
    """窗口客户区在屏幕上的 (x, y, w, h)；没有这个窗口返回 None。"""
    try:
        out = subprocess.run(["xwininfo", "-name", name], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return None
    v = {k: int(m.group(1)) for k, pat in (("x", r"Absolute upper-left X:\s+(-?\d+)"),
                                           ("y", r"Absolute upper-left Y:\s+(-?\d+)"),
                                           ("w", r"Width:\s+(\d+)"), ("h", r"Height:\s+(\d+)"))
         if (m := re.search(pat, out))}
    if len(v) < 4 or v["w"] < 50 or v["h"] < 50 or "IsViewable" not in out:
        return None
    return v["x"], v["y"], v["w"], v["h"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="RobotDecision closed loop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--wait", type=float, default=600.0, help="最多等窗口出现多少秒")
    ap.add_argument("--max", type=float, default=1800.0, help="最多录多少秒")
    ap.add_argument("--pad-bottom", type=int, default=40,
                    help="往下多录多少像素：xwininfo 给的起点含标题栏，不多录会把窗口底部那行文字截掉")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    g = None
    while g is None and time.time() - t0 < args.wait:
        g = geometry(args.name)
        if g is None:
            time.sleep(0.5)
    if g is None:
        print(f"等了 {args.wait:.0f}s 没等到窗口「{args.name}」", flush=True)
        return 1
    W, H = g[2] // 2 * 2, (g[3] + args.pad_bottom) // 2 * 2
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    print(f"开始录：窗口 {g[2]}x{g[3]} @ ({g[0]},{g[1]}) → {out}", flush=True)
    dt = 1.0 / args.fps
    t_start, n, last_seen, last_geo = time.time(), 0, time.time(), time.time()
    frame = None
    while time.time() - t_start < args.max:
        now = time.time()
        if now - last_geo > 2.0:
            g2 = geometry(args.name)
            last_geo = now
            if g2 is not None:
                g, last_seen = g2, now
            elif now - last_seen > 3.0:
                break                                          # 窗口关了
        try:
            im = ImageGrab.grab(bbox=(g[0], g[1], g[0] + g[2], g[1] + g[3] + args.pad_bottom), xdisplay=":0")
            frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
        except Exception:  # noqa: BLE001
            pass
        # 按真实时间补帧：截屏慢了就重复上一帧，保证视频时长 = 真实时长
        while frame is not None and n * dt <= time.time() - t_start:
            vw.write(frame)
            n += 1
        time.sleep(max(0.0, t_start + n * dt - time.time()))
    vw.release()
    print(f"录完：{n} 帧、{n * dt:.0f} 秒 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
