#!/usr/bin/env python3
"""在套筒正上方走水平网格、逐点拍三路相机 —— 标定腕部画面方向 + 补四个方向的近处评测数据。

为什么要这一步：示范轨迹在离套筒 4cm 内基本是一条线（边降边靠），x / y / z 一起变，
拿它没法分开标「手往哪个方向走、套筒在腕部画面里往哪边移」。网格把 x、y 拆开了。

流程（运动和到位检查都复用 replay_capture.py：MoveAbsJ 逐点、实测关节到位 <0.3° 才拍）
  A  白板上不放东西，沿示范走到抓取位姿（不拍），手张开
  ——  停下等人：把套筒放进拇指和食指之间，回车
  B  竖直抬到 +25 → +50mm；在 +50 层走 5×5 网格（±20mm、间距 10mm，蛇形），逐点拍
  C  回中心、抬到 +80mm，同样的网格再走一遍
  D  回中心，降到示范路径上离得最近的那个点（高 ≥ +30mm），沿示范倒退回起点（不拍）
姿态全程保持抓取时的朝向；网格点的关节角用自己的 PoE 逆解（以上一点为种子），
--dry-run 时就全部算好并检查：逆解是否成功、FK 回代误差、软限位余量、相邻点关节跳变。

    python grid_capture.py --dry-run
    python grid_capture.py --mock
    python grid_capture.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_capture as rc  # noqa: E402

LAYERS = (50.0, 80.0)            # mm，相对抓取位姿的法兰高度
SPAN, STEP = 20.0, 10.0          # 网格 ±20mm、间距 10mm → 5×5
MAX_JUMP_DEG = 8.0


def grid_offsets():
    """蛇形：相邻两点只差一格。"""
    vals = np.arange(-SPAN, SPAN + 0.1, STEP)
    pts = []
    for j, dy in enumerate(vals):
        xs = vals if j % 2 == 0 else vals[::-1]
        pts += [(dx, dy) for dx in xs]
    return pts


def build_grid_plan(plan):
    from ar5_poe_ik import AR5PoeIK

    # 默认 max_step=0.0148rad：那是给 125Hz 遥操防跳的「每次调用最多转 0.85°」，规划时会让解只走一小段
    # （dry-run 实测回代差 3~41mm）。这里放开，跳变由下面的 MAX_JUMP_DEG 检查管。
    ik = AR5PoeIK(max_step=np.pi, refine_iters=10)
    q, T = plan["q"], plan["T"]
    g = plan["i_grasp"]
    Tg = T[g]
    steps = []          # (标签, dx, dy, dz, 是否拍)

    def add(label, dx, dy, dz, shoot):
        steps.append((label, dx, dy, dz, shoot))

    add("lift", 0, 0, 25, False)
    for li, dz in enumerate(LAYERS):
        if li == 0:
            add("lift", 0, 0, dz, False)
        else:
            add("center", 0, 0, LAYERS[li - 1], False)
            add("lift", 0, 0, dz, False)
        # 从中心走到网格起点（-SPAN,-SPAN）：一格一格走，不跨大步
        for k in range(1, int(SPAN / STEP) + 1):
            add("to_corner", -k * STEP, -k * STEP, dz, False)
        for dx, dy in grid_offsets():
            add(f"G{int(dz)}", dx, dy, dz, True)
        # 网格终点 → 中心（同样一格一格）
        last = grid_offsets()[-1]
        n = int(round(max(abs(last[0]), abs(last[1])) / STEP))
        for k in range(n - 1, -1, -1):
            add("to_center", last[0] * k / n, last[1] * k / n, dz, False)

    # 退回点：示范路径上「比抓取高 ≥ +30mm」里水平离抓取最近的点。先在最高层水平移到它正上方，再下降过去
    P = plan["P"]
    cand = [i for i in plan["shoot"] if P[i, 2] - P[g, 2] >= 30.0 and i < g]
    j0 = min(cand, key=lambda i: np.linalg.norm(P[i, :2] - P[g, :2]))
    dxy = P[j0, :2] - P[g, :2]
    add("pre_retreat", float(dxy[0]), float(dxy[1]), LAYERS[-1], False)

    qs, problems = [], []
    seed = q[g].copy()
    lo = np.asarray(rc_cfg_arm().joint_min) + rc_cfg_arm().joint_limit_margin
    hi = np.asarray(rc_cfg_arm().joint_max) - rc_cfg_arm().joint_limit_margin
    for label, dx, dy, dz, shoot in steps:
        Tt = Tg.copy()
        Tt[:3, 3] = Tg[:3, 3] + np.array([dx, dy, dz]) / 1000.0
        sol, info = ik.ik(Tt, seed)
        if sol is None:
            problems.append(f"{label} ({dx:+.0f},{dy:+.0f},{dz:+.0f}) 逆解失败：{info.get('reason')}")
            qs.append(seed.copy())
            continue
        Tc = ik.cal.fk(sol)
        perr = np.linalg.norm(Tc[:3, 3] - Tt[:3, 3]) * 1000
        rerr = np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3].T @ Tt[:3, :3]) - 1) / 2, -1, 1)))
        if perr > 1.0 or rerr > 0.5:
            problems.append(f"{label} ({dx:+.0f},{dy:+.0f},{dz:+.0f}) FK 回代差 {perr:.1f}mm/{rerr:.2f}°")
        if np.any(sol < lo) or np.any(sol > hi):
            problems.append(f"{label} ({dx:+.0f},{dy:+.0f},{dz:+.0f}) 超软限位")
        jump = np.degrees(np.abs(sol - seed)).max()
        if jump > MAX_JUMP_DEG:
            problems.append(f"{label} ({dx:+.0f},{dy:+.0f},{dz:+.0f}) 关节跳 {jump:.1f}°")
        qs.append(sol)
        seed = sol

    jump = np.degrees(np.abs(q[j0] - qs[-1])).max()
    if jump > 2 * MAX_JUMP_DEG:
        problems.append(f"退回第一步关节跳 {jump:.1f}°")
    retreat = [i for i in reversed(plan["transit"]) if i <= j0]
    return {"steps": steps, "qs": np.array(qs), "problems": problems, "retreat_from": j0,
            "retreat": retreat, "Tg": Tg}


def rc_cfg_arm():
    from atom_robot.ar5.config import AR5Config
    return AR5Config(mock=True).arm


def print_grid(plan, gp):
    P, g = plan["P"], plan["i_grasp"]
    n_shoot = sum(s[4] for s in gp["steps"])
    print(f"抓取位姿 法兰 ({P[g,0]:.0f},{P[g,1]:.0f},{P[g,2]:.0f})；网格层 {LAYERS} mm，±{SPAN:.0f}mm 间距 {STEP:.0f}mm")
    print(f"  网格动作 {len(gp['steps'])} 步，其中拍照 {n_shoot} 个点")
    Q = gp["qs"]
    jumps = np.degrees(np.abs(np.diff(Q, axis=0))).max(1)
    print(f"  相邻两步关节最大差 {jumps.max():.1f}°（中位 {np.median(jumps):.1f}°）")
    j0 = gp["retreat_from"]
    print(f"  退回：降到示范点 t={plan['t'][j0]:.1f}s 法兰 ({P[j0,0]:.0f},{P[j0,1]:.0f},{P[j0,2]:.0f})"
          f"（比抓取高 {P[j0,2]-P[g,2]:.0f}mm、水平差 {np.linalg.norm(P[j0,:2]-P[g,:2]):.0f}mm），"
          f"再沿示范倒退 {len(gp['retreat'])} 点回起点")
    est = len(plan["shoot"]) + len(gp["steps"]) + len(gp["retreat"])
    print(f"  总共约 {est} 步 MoveAbsJ，估计 {est * 1.3 / 60:.0f}~{est * 2 / 60:.0f} 分钟")


def run(args):
    plan = rc.make_plan(args)
    gp = build_grid_plan(plan)
    print_grid(plan, gp)
    for p in gp["problems"] + rc.check_plan(plan, rc_cfg_arm()):
        print(f"  ✗ {p}")
    if args.dry_run:
        print("\n--dry-run：只算了计划，没连硬件。")
        return 0 if not gp["problems"] else 1
    if gp["problems"]:
        print("计划没通过检查，不跑。")
        return 1

    # 把网格点接到 plan 后面，复用 Rig.shoot（它按 plan 的下标取目标关节角）
    n_demo = len(plan["q"])
    plan["q"] = np.vstack([plan["q"], gp["qs"]])
    plan["t"] = np.append(plan["t"], np.full(len(gp["qs"]), -1.0))

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    for sub in ("top", "wrist", "side"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"\n输出 → {out}")
    if not args.mock:
        print("=" * 66)
        print("  ⚠ 臂会沿示范走到抓取位姿，等你放好套筒后在套筒上方 5cm / 8cm 走 5×5 网格。")
        print("    开始前白板上**什么都不要放**，臂在起始位。会自己跑完；只有急停才按 q。")
        print("    抬到 +5cm 后请看一眼：指尖要高过套筒顶；不是的话按 q。")
        print("=" * 66)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    rig = rc.Rig(args, out)
    q = plan["q"]
    try:
        rig.connect()
        diff = np.degrees(np.abs(rig.joints() - q[0])).max()
        if diff > 3.0 and not args.mock:
            raise RuntimeError(f"臂不在示范起点（最大差 {diff:.1f}°）。先 python home_arm.py 回位再跑")
        with rc.Keys(mock=args.mock) as keys:
            print(f"手 → 开：{rig.set_hand('open')}")
            print("\nA 走到抓取位姿（不拍）")
            for i in plan["transit"][1:]:
                rig.move(q[i], keys)
                print(f"\r  示范 t={plan['t'][i]:5.1f}s", end="", flush=True)
            ans = keys.ask("\n\n  把蓝套筒放进拇指和食指之间（摆成示范时那样），放好回车；输入 q 中止: ")
            if ans == "q":
                raise rc.StopRequested("放套筒时选择中止")
            rig.sleeve = True

            print("\nB/C 网格")
            Gp = plan["P"][plan["i_grasp"]]
            for k, (label, dx, dy, dz, shoot) in enumerate(gp["steps"]):
                rig.move(gp["qs"][k], keys)
                if shoot:
                    rig.shoot(label, n_demo + k, plan)
                    rig.rows[-1].update(dx=dx, dy=dy, dz=dz, gx=Gp[0], gy=Gp[1], gz=Gp[2])
                else:
                    print(f"\r  {label} ({dx:+.0f},{dy:+.0f},{dz:+.0f})            ", end="", flush=True)

            print("\n\nD 退回起点（不拍）")
            for i in gp["retreat"]:
                rig.move(q[i], keys)
                print(f"\r  示范 t={plan['t'][i]:5.1f}s", end="", flush=True)
            print("\n\n完成，臂回到示范起点。套筒还在白板上。")
    except rc.StopRequested as e:
        print(f"\n\n停下：{e}。臂停在原地，手保持张开。")
    except Exception as e:  # noqa: BLE001
        print(f"\n\n出错停下：{type(e).__name__}: {e}。臂停在原地，手保持张开。")
    finally:
        for r in rig.rows:
            for k in ("dx", "dy", "dz", "gx", "gy", "gz"):
                r.setdefault(k, "")
        rig.save_meta(plan)
        print(f"已拍 {len(rig.rows)} 组 → {out}")
        rig.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default=str(rc.DEFAULT_TRAJ))
    ap.add_argument("--t-grasp", type=float, default=rc.DEFAULT_T_GRASP)
    ap.add_argument("--t-release", type=float, default=rc.DEFAULT_T_RELEASE)
    ap.add_argument("--step-mm", type=float, default=15.0)
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
