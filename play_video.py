#!/usr/bin/env python3
"""本机没装视频播放器，用 OpenCV 窗口放 mp4：空格 暂停/继续，← / → 退 / 进 5 秒，q 或 Esc 退出，放完从头循环。

    python play_video.py ~/atom_episodes/robotdecision_auto/20261014_093856/interface.mp4
"""
import sys

import cv2

path = sys.argv[1]
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    sys.exit(f"打不开 {path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
win = "video: " + path.rstrip("/").split("/")[-2]
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))   # 默认开得很小
paused, frame = False, None
while True:
    if not paused or frame is None:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = f
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    show = frame.copy()
    cv2.putText(show, f"{pos / fps:5.1f}s / {n / fps:.0f}s  {'PAUSED' if paused else ''}", (show.shape[1] - 260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(win, show)
    k = cv2.waitKeyEx(int(1000 / fps))
    if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
        break
    if k == ord(" "):
        paused = not paused
    elif k in (65361, 2424832):                      # ←
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - int(5 * fps))); frame = None
    elif k in (65363, 2555904):                      # →
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, pos + int(5 * fps))); frame = None
cap.release()
cv2.destroyAllWindows()
