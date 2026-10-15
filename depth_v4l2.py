#!/usr/bin/env python3
"""用纯 V4L2 读 Orbbec 的深度节点（Z16），后台常驻 v4l2-ctl、线程按整帧读、只留最新一帧。

10-14 实测：两台的 :1.0 接口 index0 都是 'Z16 ' 深度，640x480 能出流（335 有效 97%、中位 628mm；305 有效 60%、中位 292mm）。
走这条路是因为：OpenCV 的 OBSENSOR 后端在 305 上 0 帧/卡死过、官方 SDK 也卡死过（见记忆 orbbec-depth-node-hangs），
而 OpenCV 的 V4L2 后端不认 Z16。纯 V4L2 和现在读彩色用的是同一个后端，不会互相污染。
注意：这样读到的是深度传感器自己的视角，**没有对齐到彩色图**。单位：335 = 1mm，305 = 0.1mm。

    python depth_v4l2.py          # 两台深度 + 彩色同时开 10 秒，报帧率、存彩色化深度图
"""
from __future__ import annotations

import subprocess
import threading
import time

import cv2
import numpy as np

DEPTH_TOP = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.0-video-index0"     # Gemini 335，1mm/单位
DEPTH_WRIST = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.0-video-index0"   # Gemini 305，0.1mm/单位
W, H = 640, 480


class DepthStream:
    def __init__(self, node, mm_per_unit, fps=15):
        self.node, self.scale, self.fps = node, float(mm_per_unit), fps
        self.proc, self.thread = None, None
        self.latest, self.t_latest, self.frames = None, 0.0, 0
        self._stop = False

    def start(self):
        cmd = ["v4l2-ctl", "-d", self.node, f"--set-fmt-video=width={W},height={H},pixelformat=Z16 ",
               f"--set-parm={self.fps}", "--stream-mmap=4", "--stream-to=-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.time()
        while self.latest is None and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if self.latest is None:
            self.stop()
            raise RuntimeError(f"深度节点 {self.node} 5 秒内没出帧")
        return self

    def _loop(self):
        n = W * H * 2
        f = self.proc.stdout
        while not self._stop:
            buf, got = bytearray(n), 0
            while got < n:
                chunk = f.read(n - got)
                if not chunk:
                    return
                buf[got:got + len(chunk)] = chunk
                got += len(chunk)
            self.latest = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(H, W)
            self.t_latest = time.time()
            self.frames += 1

    def get_mm(self, max_age=1.0):
        """最新一帧（mm，float32，0 = 无效）；太旧就返回 None。"""
        if self.latest is None or time.time() - self.t_latest > max_age:
            return None
        return self.latest.astype(np.float32) * self.scale

    def stop(self):
        self._stop = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def colorize(mm, near, far, label=""):
    """固定量程上色：近 = 红、远 = 蓝，无效 = 黑；底部写量程，方便模型读。"""
    x = np.clip((far - mm) / (far - near), 0, 1)                   # 近 → 1 → 红
    img = cv2.applyColorMap((255 * x).astype(np.uint8), cv2.COLORMAP_JET)
    img[mm <= 0] = 0
    cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
    cv2.putText(img, f"{label} depth: red = {near / 10:.0f} cm (near), blue = {far / 10:.0f} cm (far), black = no data",
                (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    top = DepthStream(DEPTH_TOP, 1.0).start()
    wr = DepthStream(DEPTH_WRIST, 0.1).start()
    cols = {k: cv2.VideoCapture(p, cv2.CAP_V4L2) for k, p in
            (("top", "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.4-video-index0"),
             ("wrist", "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.4-video-index0"))}
    t0, n = time.time(), {k: 0 for k in cols}
    f0 = {"top": top.frames, "wr": wr.frames}
    while time.time() - t0 < 10:
        for k, c in cols.items():
            ok, _ = c.read()
            n[k] += ok
    dt = time.time() - t0
    print(f"10 秒内：335 深度 {(top.frames - f0['top']) / dt:.1f} 帧/秒，305 深度 {(wr.frames - f0['wr']) / dt:.1f} 帧/秒，"
          f"335 彩色 {n['top'] / dt:.1f}，305 彩色 {n['wrist'] / dt:.1f}")
    for name, s, near, far in (("top", top, 350, 750), ("wrist", wr, 50, 450)):
        mm = s.get_mm()
        v = mm[mm > 0]
        print(f"{name}：有效 {v.size / mm.size:.0%}，中位 {np.median(v):.0f}mm")
        cv2.imwrite(f"{out}/depth_{name}.jpg", colorize(mm, near, far, name))
    for c in cols.values():
        c.release()
    top.stop(), wr.stop()
