#!/usr/bin/env python3
"""末端三个旋转轴各转一下给人看：哪个轴会改变「哪根手指最低」。

起始位竖直抬到 z340 → 平移到 --spot 上方、抬到 --z（默认 400，指尖离板 10cm 以上）→
依次绕法兰自己的 x、y、z 轴（法兰原点不动）往正、负各转到 ±--deg（每 1° 一步），到头拍三路、停下等人看完回车，再转回来。
每一小步都过逆解检查（可达、回代、软限位、单步关节 ≤10°）；某个方向转不到 --deg 就只转到能转的最大角度。
最后原路回起始位。白板上不用放东西；q 急停。

    python axis_look.py --dry-run
    python axis_look.py --mock
    python axis_look.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_capture as rc  # noqa: E402
import closed_loop as CL  # noqa: E402
from calib_start import LIFT_Z, MOVE_STEP, blob  # noqa: E402

ROT_STEP = 1.0                   # 1° 一步：起始姿态 j7 离软限位只有 6°，粗步长会把能转的范围吃掉
AXES = (("x", 0), ("y", 1), ("z", 2))


def line(kin, q, p0, p1, R, seq):
    """直线小步走 p0→p1（姿态 R 不变），追加到 seq，返回最后的 q。"""
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / MOVE_STEP)))
    for k in range(1, n + 1):
        q, why = kin.solve(q, p0 + (p1 - p0) * k / n, R, -1e9)
        if q is None:
            raise RuntimeError(f"走不过去：{why}")
        seq.append(q)
    return q


def turn(kin, q, p, R0, axis, deg, pivot=None):
    """绕法兰自己的第 axis 根轴转到 deg（每 1° 一步），转动中心 pivot（基座 mm，默认法兰原点）不动。
    返回（关节序列, 实际转到的角度, 卡住的原因）。"""
    e = np.eye(3)[axis]
    c = np.asarray(p if pivot is None else pivot, float)
    seq, done = [], 0.0
    for k in range(1, int(round(abs(deg) / ROT_STEP)) + 1):
        a = np.sign(deg) * ROT_STEP * k
        Rk = R0 @ CL.rot(e, np.radians(a))
        pk = c + Rk @ R0.T @ (np.asarray(p, float) - c)          # 法兰绕 pivot 跟着转
        qk, why = kin.solve(q, pk, Rk, -1e9)
        if qk is None:
            return seq, done, why
        seq.append(qk)
        q, done = qk, a
    return seq, done, ""


def plan(kin, q_home, R0, p_home, spot, z, deg, pivot_off=(0, 0, 0)):
    lift = []
    q = line(kin, q_home, p_home, np.array([p_home[0], p_home[1], LIFT_Z]), R0, lift)
    q = line(kin, q, np.array([p_home[0], p_home[1], LIFT_Z]), np.array([spot[0], spot[1], LIFT_Z]), R0, lift)
    p = np.array([spot[0], spot[1], z])
    q = line(kin, q, np.array([spot[0], spot[1], LIFT_Z]), p, R0, lift)
    turns = []
    for name, ax in AXES:
        for sgn in (1, -1):
            seq, done, why = turn(kin, q, p, R0, ax, sgn * deg, p + np.asarray(pivot_off, float))
            turns.append({"axis": name, "ax": ax, "target": sgn * deg, "done": done, "why": why, "seq": seq})
    return lift, p, turns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", type=float, nargs=2, default=(340.0, -160.0))
    ap.add_argument("--z", type=float, default=400.0)
    ap.add_argument("--deg", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_axis")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--auto", action="store_true", help="不等人：不用输入 yes、每个姿态停 --hold 秒后自己转回来（没有终端时用）")
    ap.add_argument("--hold", type=float, default=6.0)
    ap.add_argument("--pivot", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="转动中心相对法兰的偏移（基座 mm，起始朝向下）。默认 0 = 绕法兰原点转")
    ap.add_argument("--no-sleeve-below", action="store_true", help="腕部画面中间看到套筒就不转（手放低时防碰）")
    ap.add_argument("--only", nargs="*", default=None, help="只转这几个方向，如 z+ y-")
    ap.add_argument("--photo-every", type=int, default=0, help="转动途中每隔几度拍一次（0 = 只在转到头时拍）")
    args = ap.parse_args()

    from experiments.robot.ar5.config import AR5Config
    kin = CL.Kin()
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    T0 = kin.fk(q_home)
    R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
    lift, p, turns = plan(kin, q_home, R0, p_home, args.spot, args.z, args.deg, args.pivot)
    print(f"法兰停在 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})，起始朝向；法兰轴在基座里："
          + "；".join(f"{n} {R0[:, a].round(2)}" for n, a in AXES))
    for t in turns:
        Q = np.vstack(t["seq"]) if t["seq"] else None
        mg = np.degrees(np.minimum(Q - kin.lo, kin.hi - Q)).min() if Q is not None else float("nan")
        tip = f"（到 {t['done'] + np.sign(t['target']) * ROT_STEP:+.0f}° 时：{t['why']}）" if t["why"] else ""
        print(f"  绕 {t['axis']} 轴 {t['target']:+.0f}°：能转到 {t['done']:+.0f}°，软限位余量 {mg:.1f}°{tip}")
    if args.dry_run:
        print("--dry-run：只算了路径，没连硬件。")
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    if not args.mock and not args.auto:
        print("=" * 66)
        print(f"  ⚠ 臂会抬到白板中间上方 z{args.z:.0f}，绕末端 x、y、z 轴各正反转一次（最多 ±{args.deg:.0f}°），")
        print("    每个转到头停下等你看完回车。白板上不用放东西，臂在起始位。q 急停。")
        print("=" * 66)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0
    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    log = []
    try:
        rig.connect()
        q_now = rig.joints()
        T_now = kin.fk(q_now)
        p_now = T_now[:3, 3] * 1000
        off = np.degrees(np.arccos(np.clip((np.trace(T_now[:3, :3].T @ R0) - 1) / 2, -1, 1)))
        with rc.Keys(mock=args.mock) as keys:
            rig.set_hand("open")
            if np.degrees(np.abs(q_now - q_home)).max() <= 3.0 or args.mock:
                print("\n从起始位抬起、移过去 ...")
                for q in lift:
                    rig.move(q, keys)
            elif off < 3.0 and p_now[2] >= LIFT_Z - 1.0:
                # 上次标定中途停下时就是这样：起始朝向、在白板上方。直线移过去，到了再按实际关节角重新规划转动
                print(f"\n臂不在起始位，但是起始朝向、在白板上方 ({p_now[0]:.0f},{p_now[1]:.0f},{p_now[2]:.0f})：直线移过去 ...")
                seq = []
                line(kin, q_now, p_now, p, R0, seq)
                for q in seq:
                    rig.move(q, keys)
            else:
                raise RuntimeError(f"臂既不在起始位，也不是起始朝向的高位（法兰 z={p_now[2]:.0f}、朝向差 {off:.0f}°）。"
                                   "先回位（灵巧手遥操/VR设备/home_arm.py）再跑")
            # 7 个关节有冗余：从别处移过来，同一末端位姿的关节角会和规划的差几度（10-13 实测差 4°）。
            # 所以转动按到位后的实际关节角现场重新规划，回程也按末端坐标直线走，不复用规划的关节序列。
            q_base = rig.joints()
            turns = []
            for name, ax in AXES:
                for sgn in (1, -1):
                    seq_t, done, why = turn(kin, q_base, p, R0, ax, sgn * args.deg, p + np.asarray(args.pivot))
                    turns.append({"axis": name, "ax": ax, "target": sgn * args.deg, "done": done, "why": why,
                                  "seq": seq_t})
            print("按实际关节角重新规划：" + "，".join(f"{t['axis']}{t['done']:+.0f}°" for t in turns))
            fr = CL.grab(rig)
            for k, im in fr.items():
                cv2.imwrite(str(out / f"00_level_{k}.jpg"), im)
            if args.no_sleeve_below and not args.mock:
                b = blob(fr.get("cam_front"))
                if b is not None and b[3] > 3000 and abs(b[0] - 320) < 200 and abs(b[1] - 240) < 170:
                    raise RuntimeError(f"腕部画面中间有套筒（面积 {b[3]:.0f}px），它可能就在手指下面，转动会碰到。"
                                       "请把套筒挪开 15cm 以上再跑")
            if args.only:
                turns = [t for t in turns if f"{t['axis']}{'+' if t['target'] > 0 else '-'}" in args.only]
            for i, t in enumerate(turns, 1):
                if not t["seq"]:
                    print(f"\n绕 {t['axis']} 轴 {t['target']:+.0f}°：一步都转不了（{t['why']}），跳过")
                    continue
                print(f"\n绕 {t['axis']} 轴 → {t['done']:+.0f}°")
                for k, q in enumerate(t["seq"], 1):
                    rig.move(q, keys)
                    a = int(round(np.sign(t["target"]) * k * ROT_STEP))
                    if args.photo_every and a % args.photo_every == 0 and k < len(t["seq"]):
                        fr = CL.grab(rig)               # 扫描：途中每隔几度拍一次
                        for kk, im in fr.items():
                            cv2.imwrite(str(out / f"{i:02d}_{t['axis']}{a:+d}_{kk}.jpg"), im)
                        print(f"  {t['axis']}{a:+d}° 拍了", flush=True)
                fr = CL.grab(rig)
                tag = f"{i:02d}_{t['axis']}{t['done']:+.0f}"
                for k, im in fr.items():
                    cv2.imwrite(str(out / f"{tag}_{k}.jpg"), im)
                if not args.mock and not args.auto:
                    CL.show([fr.get("cam_high"), fr.get("cam_front"), fr.get("cam_side")],
                            f"rotate about flange {t['axis']} {t['done']:+.0f} deg")
                log.append({"axis": t["axis"], "deg": t["done"], "limit": t["why"], "tag": tag})
                if args.auto:
                    print(f"  绕末端 {t['axis']} 轴转了 {t['done']:+.0f}°，停 {args.hold:.0f} 秒", flush=True)
                    time.sleep(0 if args.mock else args.hold)
                elif keys.ask(f"  绕末端 {t['axis']} 轴转了 {t['done']:+.0f}°。看好了回车转回来，q 停: ") == "q":
                    raise rc.StopRequested("看姿态时停下")
                for q in reversed(t["seq"][:-1]):
                    rig.move(q, keys)
                rig.move(q_base, keys)
            print("\n回起始位 ...")
            back, q = [], q_base
            q = line(kin, q, p, np.array([p[0], p[1], LIFT_Z]), R0, back)
            q = line(kin, q, np.array([p[0], p[1], LIFT_Z]), np.array([p_home[0], p_home[1], LIFT_Z]), R0, back)
            q = line(kin, q, np.array([p_home[0], p_home[1], LIFT_Z]), p_home, R0, back)
            gap = np.degrees(np.abs(q - q_home)).max()
            if gap > 10.0:
                raise RuntimeError(f"走回起始位上方后关节和起始位差 {gap:.0f}°，不直接甩过去，停在这里")
            for q in back:
                rig.move(q, keys)
            rig.move(q_home, keys)
            print("已回到起始位。")
    except rc.StopRequested as e:
        print(f"\n停下：{e}。臂停在原地。")
    except Exception as e:  # noqa: BLE001
        print(f"\n出错停下：{type(e).__name__}: {e}。臂停在原地。")
    finally:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"照片 → {out}")
        cv2.destroyAllWindows()
        rig.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
