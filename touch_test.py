#!/usr/bin/env python3
"""两件事：① 手在空中合拢时拇指食指尖碰在腕部画面的哪里（瞄准点，和手的朝向无关）；② 指尖碰板能不能靠手指受力读数测出来。

10-13 人手推指尖实测：静止受力只抖 ±2，推一下跳几百（正负都有），角度几乎不变 → 用受力偏离判断碰到。
这里验证臂往下降、指尖碰到泡沫白板时也一样清楚，并核对碰板高度和人看着量的下限（291.9，指尖离板 3~5mm）对得上。

流程（起始朝向不变；每一小步逆解检查后 MoveAbsJ，实测到位才继续）
  起始位竖直抬到 z340 → 移到 --spot 上方 → 法兰 下限+90
  A 合手拍腕部照片 → 张手
  B 每 10mm 降到 下限+15（途中记受力，看运动本身会不会误触发）→ 取静止受力 →
    每 2mm 往下降，六路受力任何一路偏离超过 --thresh 就停；法兰最低不低于 下限−8
  抬回 下限+90 → 回起始位。白板上 --spot 那块不能放东西。

    python touch_test.py --dry-run
    python touch_test.py --mock
    python touch_test.py --auto      # 不等人确认（没有终端时用）
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
from calib_start import LIFT_Z, Mover  # noqa: E402

FLOOR = 291.9            # 10-13 第二次标定人看着量的：起始朝向下最低指尖离板 3~5mm 时的法兰高度
NAMES = ["小指", "无名指", "中指", "食指", "拇指弯", "拇指转"]


def forces(hand, n=3):
    """读 n 次取中位（单次偶尔跳）。"""
    return np.median(np.array([hand.get_force() for _ in range(n)], float), axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", type=float, nargs=2, default=(340.0, -160.0))
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--thresh", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--max-below", type=float, default=8.0, help="最多比下限再低多少 mm")
    ap.add_argument("--others", type=int, default=950,
                    help="中指/无名指/小指的张开指令。10-13：1000 顶在机械限位上，指尖碰板的力被限位挡住、电机读不到，改 950 离开限位")
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_touch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--auto", action="store_true")
    args = ap.parse_args()

    from atom_robot.ar5.config import AR5Config
    kin = CL.Kin()
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    T0 = kin.fk(q_home)
    R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
    s = np.asarray(args.spot, float)
    z_hi, z_slow, z_min = args.floor + 90, args.floor + 15, args.floor - args.max_below
    # 离线演算：整条路径能不能走
    m = Mover(None, kin, None, R0, q_home)
    for p in ([p_home[0], p_home[1], LIFT_Z], [s[0], s[1], LIFT_Z], [s[0], s[1], z_hi], [s[0], s[1], z_min],
              [s[0], s[1], z_hi], [s[0], s[1], LIFT_Z], [p_home[0], p_home[1], LIFT_Z], p_home):
        m.go(p)
    Q = np.vstack(m.trace)
    print(f"标定点 ({s[0]:.0f},{s[1]:.0f})：法兰 {z_hi:.0f} 拍合手 → 每 10mm 降到 {z_slow:.0f} → 每 {args.step:g}mm 降，"
          f"最低 {z_min:.0f}；受力偏离 > {args.thresh:g} 停")
    print(f"  全程 {len(Q)} 步可走，单步关节最多 {np.degrees(np.abs(np.diff(Q, axis=0))).max():.1f}°，"
          f"最小软限位余量 {np.degrees(np.minimum(Q - kin.lo, kin.hi - Q)).min():.1f}°")
    if args.dry_run:
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    if not args.mock and not args.auto:
        if input("  臂会移到白板中间、合一次手、再慢慢降到指尖碰板。确认后输入 yes 回车: ").strip().lower() != "yes":
            return 0
    rc.OTHERS_OPEN = int(args.others)          # Rig.set_hand 用的是这个模块常量
    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    log = {"floor": args.floor, "thresh": args.thresh, "others": args.others, "descent": [], "touch": None}

    def snap(tag):
        fr = CL.grab(rig)
        for k, im in fr.items():
            cv2.imwrite(str(out / f"{tag}_{k}.jpg"), im)
        return fr

    try:
        rig.connect()
        q_now = rig.joints()
        if np.degrees(np.abs(q_now - q_home)).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在起始位。先回位（灵巧手遥操/VR设备/home_arm.py）再跑")
        with rc.Keys(mock=args.mock) as keys:
            mv = Mover(rig, kin, keys, R0, q_now)
            rig.set_hand("open")
            print("\n抬起、移过去 ...", flush=True)
            mv.go([p_home[0], p_home[1], LIFT_Z])
            mv.go([s[0], s[1], LIFT_Z])
            mv.go([s[0], s[1], z_hi])
            snap("open_hi")

            print("A 合手拍腕部照片", flush=True)
            ang = rig.set_hand("close")
            time.sleep(0.5)
            snap("closed_hi")
            log["closed_angles"] = [int(a) for a in ang]
            rig.set_hand("open")
            time.sleep(0.5)

            print(f"B 每 10mm 降到 {z_slow:.0f}（记受力看运动会不会误触发）", flush=True)
            f_air = [forces(rig.hand)]
            z = z_hi
            while z - 10 >= z_slow:
                z -= 10
                mv.go([s[0], s[1], z])
                f = forces(rig.hand)
                f_air.append(f)
                log["descent"].append({"z": z, "f": f.tolist(), "phase": "air"})
            mv.go([s[0], s[1], z_slow])
            base = forces(rig.hand, 7)
            dev_air = np.abs(np.array(f_air) - base).max(0)
            print(f"  静止受力 {base.astype(int).tolist()}；降的途中最大偏离 {dev_air.astype(int).tolist()}", flush=True)
            log["base"], log["air_dev"] = base.tolist(), dev_air.tolist()

            print(f"  每 {args.step:g}mm 往下降：", flush=True)
            z = z_slow
            touched = None
            while z - args.step >= z_min - 1e-6:
                z -= args.step
                mv.go([s[0], s[1], z])
                f = forces(rig.hand)
                if args.mock and z <= args.floor - 4:          # 模拟：下限下方 4mm 碰到
                    f = base + [0, 0, 0, 0, 0, 0] + np.array([180, 60, 0, 0, 0, 0])
                d = f - base
                k = int(np.argmax(np.abs(d)))
                log["descent"].append({"z": z, "f": f.tolist(), "phase": "slow"})
                print(f"    法兰 {z:6.1f}  偏离 {d.astype(int).tolist()}  最大 {NAMES[k]} {d[k]:+.0f}", flush=True)
                if abs(d[k]) > args.thresh:
                    touched = {"z": z, "finger": NAMES[k], "dev": d.tolist()}
                    break
            if touched:
                fr = snap("touch")
                log["touch"] = touched
                print(f"  → 碰到了：法兰 {touched['z']:.1f}mm，最先反应的是{touched['finger']}"
                      f"（人量的下限 {args.floor:.1f}，差 {touched['z'] - args.floor:+.1f}mm）", flush=True)
            else:
                snap("no_touch")
                print(f"  → 降到最低 {z:.1f}mm 也没测到碰板", flush=True)

            print("\n抬起、回起始位 ...", flush=True)
            mv.go([s[0], s[1], z_hi])
            mv.go([s[0], s[1], LIFT_Z])
            mv.go([p_home[0], p_home[1], LIFT_Z])
            mv.go(p_home)
            rig.move(q_home, keys)
            print("已回到起始位。", flush=True)
    except rc.StopRequested as e:
        print(f"\n停下：{e}。臂停在原地。")
    except Exception as e:  # noqa: BLE001
        print(f"\n出错停下：{type(e).__name__}: {e}。臂停在原地。")
    finally:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"记录 → {out}")
        rig.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
