#!/usr/bin/env python3
"""从白板上方任意高位安全回起始位：保持朝向平移回 (450,0,z) → 转回起始朝向 → 起始位上方 z340 → 关节过渡到起步那组 → 原路降回。
--go 才真动；不加只算路径。"""
import sys, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_capture as rc, closed_loop as CL, auto_loop2 as L
from atom_robot.ar5.config import AR5Config
ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); args = ap.parse_args()
kin = CL.Kin(); q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
T0 = kin.fk(q_home); R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
lift = [(np.array([p_home[0], p_home[1], z]), R0) for z in list(np.arange(p_home[2] + 10, L.LIFT_Z - 0.01, 10)) + [L.LIFT_Z]]
class Fake:                        # 离线算 lift 关节用
    def __init__(s, q): s.q = q
    def joints(s): return s.q
lift_seq, why = L.Arm(Fake(q_home), kin, None).plan(lift, floor=-1e9)
rig = rc.Rig(argparse.Namespace(mock=False, speed=60.0), None)
rig.cams = rc.FakeCams()
try:
    rig.connect()                  # 相机已换成假的；连手是为了设力和速度，连臂并设负载
    arm = L.Arm(rig, kin, None)
    q, p, R = arm.now()
    print(f"现在：法兰 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})，离起始朝向 {np.degrees(CL.axis_angle(R0.T @ R)[1]):.1f}°")
    # 10-13 并行方案第一次：放回套筒后手在底座附近 (234,-287,267)，这里竖直抬不到 360（夹着时也只抬到 346）。
    # 抬升高度从高往低试，取整条回程都可行的第一个
    class Seed:
        def __init__(s, q): s.q = q
        def joints(s): return s.q
    plans = None
    for z in sorted({max(p[2], 360.0), 345.0, 330.0, 315.0, round(p[2] + 30, 1), round(float(p[2]), 1)}, reverse=True):
        if z < p[2] + (20 if p[2] < 320 else 0):     # 10-14：已在 340 高处时 360/370 够不着，允许就在当前高度出发
            continue
        segs = [("先竖直抬起（手可能紧挨着套筒，别横着扫）", [(np.array([p[0], p[1], z]), R)]),
                ("保持朝向平移回白板上方", [(np.array([450, 0, max(z, 340.0)]), R)]),
                ("转回起始朝向", [(np.array([450, 0, max(z, 340.0)]), R0)]),
                ("到起始位上方 z340", [(np.array([p_home[0], p_home[1], L.LIFT_Z]), R0)])]
        tried, qs, why = [], q, ""
        for name, path in segs:
            seq, why = L.Arm(Seed(qs), kin, None).plan(path, floor=-1e9)
            if seq is None:
                break
            tried.append((name, seq, qs)); qs = seq[-1]
        else:
            plans = tried
            print(f"  抬到 z{z:.0f}：整条回程可行")
            break
        print(f"  抬到 z{z:.0f}：{name} 走不了（{why}）")
    if plans is None:
        print("✗ 试过的高度都走不了"); sys.exit(1)
    for name, seq, q0_ in plans:
        print(f"  {name}：{len(seq)} 步，单步最多 {np.degrees(np.abs(np.diff(np.vstack([q0_, *seq]), axis=0))).max():.1f}°")
    plans = [s for _, s, _ in plans]
    gap = np.degrees(np.abs(lift_seq[-1] - qs)).max()
    n = max(1, int(np.ceil(gap / 3.0)))
    trans = [qs + (lift_seq[-1] - qs) * k / n for k in range(1, n + 1)]
    zmin = min(kin.fk(t)[2, 3] * 1000 for t in trans)
    print(f"  关节过渡到起步那组：差 {gap:.1f}°，分 {n} 步，途中法兰最低 {zmin:.0f}mm")
    if zmin < L.LIFT_Z - 40:
        print("✗ 过渡途中太低，不走"); sys.exit(1)
    print(f"  原路降回：{len(lift_seq) - 1} 步")
    if not args.go:
        print("只算了路径，没动臂。"); sys.exit(0)
    with rc.Keys(mock=False) as keys:
        for seq in plans:
            for qq in seq: rig.move(qq, keys)
        for qq in trans: rig.move(qq, keys)
        for qq in reversed(lift_seq[:-1]): rig.move(qq, keys)
        rig.move(q_home, keys)
    print("已回到起始位。")
finally:
    rig.close()
