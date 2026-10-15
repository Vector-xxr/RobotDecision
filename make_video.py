#!/usr/bin/env python3
"""把一次运行的界面画面拼成视频：窗口里显示的就是每步存下的 *_panel.jpg，按 log.json 里每步的时间排。

不录屏（10-14 试过：录屏录的是屏幕那块区域，用户在用别的软件时录进去的是 WPS），所以直接用程序自己存的画面，
内容和「RobotDecision closed loop」窗口里显示的一模一样，时长 = 真实时长（第 k 步的画面一直显示到第 k+1 步）。

    python make_video.py ~/atom_episodes/robotdecision_auto/20261014_093856
    python make_video.py <运行目录> --out xxx.mp4 --speed 2      # 2 倍速
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None, help="默认存到运行目录里的 interface.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="倍速（1 = 真实时长）")
    ap.add_argument("--hold-end", type=float, default=3.0, help="最后一张多停几秒")
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    L = json.load(open(run / "log.json"))
    log = L["log"] if isinstance(L, dict) else L
    steps = [(r["step"], float(r["t"])) for r in log if (run / f"{r['step']:03d}_panel.jpg").exists()]
    if not steps:
        print("没有找到 *_panel.jpg")
        return 1
    first = cv2.imread(str(run / f"{steps[0][0]:03d}_panel.jpg"))
    H, W = first.shape[:2]
    out = Path(args.out).expanduser() if args.out else run / "interface.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W // 2 * 2, H // 2 * 2))
    n = 0
    for i, (step, t) in enumerate(steps):
        img = cv2.imread(str(run / f"{step:03d}_panel.jpg"))
        img = cv2.resize(img, (W // 2 * 2, H // 2 * 2))
        t_next = steps[i + 1][1] if i + 1 < len(steps) else t + args.hold_end * args.speed
        k = max(1, round((t_next - t) / args.speed * args.fps))
        for _ in range(k):
            vw.write(img)
        n += k
    vw.release()
    result = L.get("result", "") if isinstance(L, dict) else ""
    print(f"{len(steps)} 步、{n / args.fps:.0f} 秒（{args.speed:g} 倍速）→ {out}    运行结果：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
