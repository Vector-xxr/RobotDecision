#!/usr/bin/env python3
"""回放一次 VR 示范的关节轨迹，逐点停稳拍俯视 + 腕部 + 侧面三张图 —— RobotDecision 离线评测的数据来源。

为什么回放而不是再遥操：示范时没录上画面，但 10Hz 轨迹（avp_servoj_teleop --traj-hz）是全的。
原样重走示范的关节角 = 走一遍人亲手开过、没撞过的路；套筒的位置由臂来定 ——
先张着手走到示范的抓取位姿，人把套筒放进拇指和食指之间，就回到了示范时的地方。

手只当二指夹爪用（用户 10-13 定）：小指/无名指/中指固定张开，拇指旋转固定，
只有食指 + 拇指弯曲在「开 / 合」两档之间切换。

流程（每一步都是控制器规划的慢速 MoveAbsJ，到位以**实测关节角**为准，不信 reached 标志）
  P1  白板上**不放**套筒。从起始位沿示范走到抓取位姿，手张开。顺路拍（= 没有目标的画面）
  ——  停下等人：把套筒放进两指之间，回车
  P2  在抓取位姿拍一张，然后沿示范**倒着**退回起始位，逐点拍（= 接近过程的画面）
  P3  再正着走回抓取位姿（只走不拍）
  P4  合手，确认手指被挡住（= 真夹住了），拍
  P5  沿示范抬起到最高点，逐点拍
  P6  沿刚才的路倒回抓取位姿 → 张手 → 拍（套筒放回原处）
  P7  沿示范倒着退回起始位（只走不拍）

    python replay_capture.py --dry-run          # 只读轨迹、打印计划和检查，不连任何硬件
    python replay_capture.py --mock             # 假臂 + 假手 + 假相机把整套流程跑一遍
    python replay_capture.py                    # 真跑（会先问 yes）

运行中按 q（急停用，正常不用按）：走完当前这一小步就停，臂停在原地、手保持当前状态（不会自己松手）。
退出时断开臂要等控制器释放会话，可能要 ~1 分钟，**别按 Ctrl+C**。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_workspace() -> Path:
    """工作区根：从本文件往上找同时有 算法/ 和 数据_硬件/ 的目录，不写死本机绝对路径。
    整个工作区搬家、换用户、拷进移动硬盘都不用改；单独拷走时用环境变量 YUHANG_WS 指过去。"""
    env = os.environ.get("YUHANG_WS")
    if env:
        return Path(env).expanduser().resolve()
    for p in HERE.parents:
        if (p / "算法").is_dir() and (p / "数据_硬件").is_dir():
            return p
    return HERE.parents[2]   # 找不到就按现在的层级（算法/Jev/DecisionAPI）推


WS = _find_workspace()
VR = WS / "数据_硬件" / "灵巧手遥操" / "VR设备"
AR5_ROOT = str(WS / "算法" / "VEL" / "vel_ar5" / "openvla-energy")
sys.path.insert(0, str(VR))
sys.path.insert(0, AR5_ROOT)

# 示教轨迹。原件在 VR设备/logs/，那里是运行日志目录，会被 git 忽略、也可能被清理，
# 所以拷了一份进本项目 data/，auto_loop 取抓取朝向和示范高度都靠它。
DEFAULT_TRAJ = HERE / "data" / "traj_20261013_101346.csv"
# 这条示范里：48.1s 手指开始被套筒挡住（指令 627 / 实测 663）= 抓取位姿；
# 57.2s 松手。抬起最高点在两者之间自动找。
DEFAULT_T_GRASP, DEFAULT_T_RELEASE = 48.1, 57.2

CAM_TOP = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:5:1.4-video-index0"     # Gemini 335 彩色
CAM_WRIST = "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:6:1.4-video-index0"   # Gemini 305 彩色
# 10-13 加的侧面 UGREEN（序列号都是 SN0001，只能按 USB 口认；单独在 10:00.4 控制器上）
CAM_SIDE = "/dev/v4l/by-path/pci-0000:10:00.4-usb-0:2:1.0-video-index0"
HAND_PORT = "tcp://192.168.11.210:6000"

LITTLE, RING, MIDDLE, INDEX, THUMB_BEND, THUMB_ROT = range(6)
OTHERS_OPEN = 1000          # 小指/无名指/中指：示范开始时实测 998/988/966，即张开
HAND_OPEN = (880, 1000)     # 食指、拇指弯曲：示范里接近套筒时的值（879 / 1000）
HAND_CLOSE = (400, 400)     # 示范夹持时的指令在 266~480；实测停在 ~652 / ~600（被套筒挡住）
THUMB_ROT_FIXED = 195       # 示范合手那几秒的拇指旋转
HAND_FORCE, HAND_SPEED = 1000, 1000   # 和遥操一样；1000 才提得起来
PAYLOAD_KG, PAYLOAD_COM_Z = 1.0, 0.08  # 和 home_arm.py 一样

ARRIVE_DEG = 0.3            # 实测和目标每个关节都差不到这么多才算到位
SETTLE_S = 0.3              # 到位后再等这么久才拍，避开残余抖动


# ── 轨迹与计划（不碰硬件） ─────────────────────────────────────────────────
def load_traj(path):
    hdr = open(path, encoding="utf-8").readline().strip().split(",")
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    col = {h: i for i, h in enumerate(hdr)}
    q = d[:, [col[f"qmeas{i}"] for i in range(1, 8)]]
    hand = {k: d[:, col[k]] for k in hdr if k.startswith(("cmd_", "meas_"))}
    return d[:, col["t"]], q, hand


def thin(q, P, idx, step_mm, step_deg):
    """沿示范抽停拍点：末端走过 step_mm 或任一关节转过 step_deg 就取一个。首尾必取。"""
    keep = [idx[0]]
    acc_p = acc_q = 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        acc_p += float(np.linalg.norm(P[b] - P[a]))
        acc_q += float(np.degrees(np.abs(q[b] - q[a])).max())
        if acc_p >= step_mm or acc_q >= step_deg:
            keep.append(b)
            acc_p = acc_q = 0.0
    if keep[-1] != idx[-1]:
        keep.append(idx[-1])
    return keep


def make_plan(args):
    from ar5_poe_ik import AR5PoeIK

    t, q, hand = load_traj(args.traj)
    fk = AR5PoeIK().cal.fk
    T = np.array([fk(x) for x in q])
    P = T[:, :3, 3] * 1000
    i_g = int(np.argmin(np.abs(t - args.t_grasp)))
    i_r = int(np.argmin(np.abs(t - args.t_release)))
    i_top = i_g + int(np.argmax(P[i_g:i_r + 1, 2]))
    approach = list(range(0, i_g + 1))
    plan = {
        "traj": str(args.traj), "t": t, "q": q, "P": P, "T": T,
        "i_grasp": i_g, "i_top": i_top,
        "shoot": thin(q, P, approach, args.step_mm, args.step_deg),             # P1/P2 拍
        # P3/P7 只走不拍，但点不抽稀：2 倍间隔时相邻点关节差到 11.3°，两点之间离示范会偏得多
        "transit": thin(q, P, approach, args.step_mm, args.step_deg),
        "lift": thin(q, P, list(range(i_g, i_top + 1)), args.step_mm, args.step_deg),
        "hand": hand,
    }
    return plan


def check_plan(plan, cfg_arm):
    q = plan["q"]
    lo = np.asarray(cfg_arm.joint_min) + cfg_arm.joint_limit_margin
    hi = np.asarray(cfg_arm.joint_max) - cfg_arm.joint_limit_margin
    problems = []
    for name in ("shoot", "transit", "lift"):
        idx = plan[name]
        Q = q[idx]
        if np.any(Q < lo) or np.any(Q > hi):
            problems.append(f"{name}: 有点在软限位外")
        jump = np.degrees(np.abs(np.diff(Q, axis=0))).max() if len(idx) > 1 else 0.0
        if jump > 8.0:
            problems.append(f"{name}: 相邻两点关节差最大 {jump:.1f}°（>8°，两点之间偏离示范可能变大）")
    if np.isnan(q).any():
        problems.append("轨迹里有缺失的关节角")
    return problems


def print_plan(plan, cfg_arm, speed):
    t, P, q = plan["t"], plan["P"], plan["q"]
    g, top = plan["i_grasp"], plan["i_top"]
    lo = np.asarray(cfg_arm.joint_min) + cfg_arm.joint_limit_margin
    hi = np.asarray(cfg_arm.joint_max) - cfg_arm.joint_limit_margin
    print(f"示范轨迹 {plan['traj']}")
    print(f"  起点 t={t[0]:.1f}s 法兰 ({P[0,0]:.0f},{P[0,1]:.0f},{P[0,2]:.0f}) mm")
    print(f"  抓取 t={t[g]:.1f}s 法兰 ({P[g,0]:.0f},{P[g,1]:.0f},{P[g,2]:.0f}) mm   "
          f"工具 z 轴 {np.round(plan['T'][g,:3,2],2).tolist()}")
    print(f"  抬起最高 t={t[top]:.1f}s 法兰 z={P[top,2]:.0f} mm（比抓取高 {P[top,2]-P[g,2]:.0f}）")
    for name, label in (("shoot", "P1/P2 拍照点"), ("transit", "P3/P7 过路点"), ("lift", "P5 抬起拍照点")):
        idx = plan[name]
        Q = q[idx]
        jump = np.degrees(np.abs(np.diff(Q, axis=0))).max() if len(idx) > 1 else 0.0
        marg = np.degrees(np.minimum(Q - lo, hi - Q)).min(0)
        print(f"  {label}: {len(idx)} 个   相邻最大关节差 {jump:.1f}°   法兰最低 z {P[idx,2].min():.0f} mm   "
              f"离软限位最小余量 j{int(np.argmin(marg))} {marg.min():.1f}°")
    n_moves = 2 * len(plan["shoot"]) + 2 * len(plan["transit"]) + 2 * len(plan["lift"])
    print(f"  一共约 {n_moves} 步 MoveAbsJ（速度参数 {speed:g}），估计 {n_moves * 1.3 / 60:.0f}~{n_moves * 2 / 60:.0f} 分钟")
    print(f"  手：小指/无名指/中指 {OTHERS_OPEN}，拇指旋转 {THUMB_ROT_FIXED}，"
          f"开 = 食指 {HAND_OPEN[0]} / 拇指弯 {HAND_OPEN[1]}，合 = {HAND_CLOSE[0]} / {HAND_CLOSE[1]}，"
          f"力 {HAND_FORCE} 速度 {HAND_SPEED}，食指-拇指耦合下限开")


# ── 硬件 ─────────────────────────────────────────────────────────────────
class StopRequested(Exception):
    pass


class Keys:
    """非阻塞读键：q = 停。提示要人回车时临时恢复成行模式。"""

    def __init__(self, mock=False):
        self.tty = sys.stdin.isatty()
        self.mock = mock
        self.old = termios.tcgetattr(sys.stdin) if self.tty else None

    def __enter__(self):
        if self.tty:
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self.tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)

    def poll(self):
        # 只认 q：回车也算停的话，放套筒时多按一下回车就会把刚开始的 P2 停掉
        if self.tty and select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch in ("q", "Q"):
                raise StopRequested(f"按了 {ch!r}")

    def ask(self, prompt):
        if not self.tty:
            if self.mock:                     # 假跑没人回答：一律「继续」
                print(prompt + "（mock：自动继续）")
                return ""
            raise StopRequested("没有终端，没法让人确认 —— 真跑必须在终端里")
        if self.tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
        try:
            return input(prompt).strip().lower()
        finally:
            if self.tty:
                termios.tcflush(sys.stdin, termios.TCIFLUSH)     # 多按的键作废
                tty.setcbreak(sys.stdin.fileno())


class FakeCams:
    def __init__(self, _cfg=None):
        pass

    def connect(self):
        pass

    def read(self):
        return {k: np.zeros((480, 640, 3), np.uint8) for k in ("cam_high", "cam_front", "cam_side")}

    def disconnect(self):
        pass


class Rig:
    def __init__(self, args, out_dir):
        from experiments.robot.ar5.ar5_env import AR5Env
        from experiments.robot.ar5.config import AR5Config
        import inspire_hand6 as ih

        self.args, self.out = args, out_dir
        cfg = AR5Config(mock=args.mock, real_hand_in_mock=False)
        cfg.use_hand = False
        cfg.arm.use_realtime = False          # 回位同款：排队式 MoveAbsJ，控制器自己规划
        cfg.arm.move_speed = float(args.speed)
        cfg.arm.move_zone = 0.0               # 每一步都要停在点上，不过渡
        cfg.cameras.sources = {}
        self.cfg = cfg
        self.env = AR5Env(cfg)
        self.ih = ih
        if args.mock:
            self.hand = ih.InspireHand6(port="tcp://mock:6000", modbus_factory=ih.MockModbus,
                                        couple_index_thumb=True)
            self.cams = FakeCams()
        else:
            from experiments.robot.ar5.drivers.cameras import OpenCVCameras
            cc = AR5Config(mock=False).cameras
            cc.sources = {"cam_high": CAM_TOP, "cam_front": CAM_WRIST, "cam_side": CAM_SIDE}
            self.hand = ih.InspireHand6(port=HAND_PORT, couple_index_thumb=True)
            self.cams = OpenCVCameras(cc)
        self.rows = []
        self.hand_state = "unknown"
        self.sleeve = False

    # 连接顺序：相机 → 手 → 臂。臂最后连，前面任何一样失败都不会让臂上电。
    def connect(self):
        print("连相机 ...", flush=True)
        self.cams.connect()
        print("连手 ...", flush=True)
        self.hand.connect()
        self.hand.set_force(HAND_FORCE)       # 力控/速度掉电不保存，每次都要设
        self.hand.set_speed(HAND_SPEED)
        if not self.args.mock:
            ok, why = self.hand.liveness()
            if not ok:
                raise RuntimeError(f"手的控制循环没在跑：{why}（给手断电重启）")
        print("连臂（不开 RT）...", flush=True)
        self.env.connect()
        r = getattr(self.env.arm, "_robot", None)
        if r is not None and hasattr(r, "setLoad"):
            ec = {}
            r.setLoad(self.env.arm.sdk.Load(PAYLOAD_KG, [0.0, 0.0, PAYLOAD_COM_Z], [0.0, 0.0, 0.0]), ec)
            print(f"  负载 {PAYLOAD_KG}kg (ec={ec.get('ec', 0)})")

    def close(self):
        for name, fn in (("相机", self.cams.disconnect), ("手", self.hand.close)):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  关{name}出错: {type(e).__name__}: {e}")
        print("断开臂（控制器释放会话可能要 ~1 分钟，别按 Ctrl+C）...", flush=True)
        try:
            self.env.close()
        except Exception as e:  # noqa: BLE001
            print(f"  断开臂出错: {type(e).__name__}: {e}")

    def joints(self):
        return np.asarray(self.env.arm.get_joint_positions(), float)[:7]

    def move(self, q_target, keys):
        keys.poll()
        self.env.goto_joints(q_target)
        t0 = time.time()
        while True:
            err = np.degrees(np.abs(self.joints() - q_target)).max()
            if err < ARRIVE_DEG:
                return
            if time.time() - t0 > 5.0:
                raise RuntimeError(f"MoveAbsJ 之后 5s 还差 {err:.2f}° —— 运动可能被控制器拒了，停下")
            time.sleep(0.05)

    def set_hand(self, state):
        idx, thumb = HAND_OPEN if state == "open" else HAND_CLOSE
        six = [OTHERS_OPEN, OTHERS_OPEN, OTHERS_OPEN, idx, thumb, THUMB_ROT_FIXED]
        self.hand.set_angles(six)
        time.sleep(1.5 if state == "close" else 1.0)
        self.hand_state = state
        return self.hand.get_angles()

    def shoot(self, phase, src_i, plan):
        time.sleep(SETTLE_S)
        frames = self.cams.read()
        q_meas = self.joints()
        n = len(self.rows)
        files = {}
        import cv2
        for key, sub in (("cam_high", "top"), ("cam_front", "wrist"), ("cam_side", "side")):
            img = frames.get(key)
            if img is None:
                continue
            fn = f"{sub}/{n:04d}.jpg"
            cv2.imwrite(str(self.out / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])        # 驱动给的是 RGB
            files[sub] = fn
        try:
            h = self.hand.get_angles()
        except Exception:  # noqa: BLE001
            h = [np.nan] * 6
        from ar5_poe_ik import AR5PoeIK
        T = plan.setdefault("_fk", AR5PoeIK().cal.fk)(q_meas)
        self.rows.append({
            "n": n, "phase": phase, "src_i": src_i, "src_t": round(float(plan["t"][src_i]), 3),
            "hand_state": self.hand_state, "sleeve_present": int(self.sleeve),
            **{f"qtgt{j+1}": float(plan["q"][src_i][j]) for j in range(7)},
            **{f"qmeas{j+1}": float(q_meas[j]) for j in range(7)},
            "x_mm": T[0, 3] * 1000, "y_mm": T[1, 3] * 1000, "z_mm": T[2, 3] * 1000,
            "zax_x": T[0, 2], "zax_y": T[1, 2], "zax_z": T[2, 2],
            **{f"hand_{nm}": h[k] for k, nm in enumerate(self.ih.DOF_NAMES)},
            "top": files.get("top", ""), "wrist": files.get("wrist", ""), "side": files.get("side", ""),
            "wall_time": time.time(),
        })
        print(f"\r  [{phase}] 第 {n:3d} 张  示范 t={plan['t'][src_i]:5.1f}s  "
              f"法兰 ({T[0,3]*1000:.0f},{T[1,3]*1000:.0f},{T[2,3]*1000:.0f})", end="", flush=True)

    def save_meta(self, plan):
        if not self.rows:
            return
        with open(self.out / "meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        info = {"traj": plan["traj"], "i_grasp": plan["i_grasp"], "i_top": plan["i_top"],
                "t_grasp": float(plan["t"][plan["i_grasp"]]), "t_top": float(plan["t"][plan["i_top"]]),
                "hand_open": HAND_OPEN, "hand_close": HAND_CLOSE, "thumb_rot": THUMB_ROT_FIXED,
                "others": OTHERS_OPEN, "force": HAND_FORCE, "speed": self.args.speed,
                "cam_top": CAM_TOP, "cam_wrist": CAM_WRIST, "cam_side": CAM_SIDE, "frames": len(self.rows),
                "mock": bool(self.args.mock)}
        (self.out / "plan.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def run(args):
    from experiments.robot.ar5.config import AR5Config

    plan = make_plan(args)
    cfg_arm = AR5Config(mock=True).arm
    print_plan(plan, cfg_arm, args.speed)
    problems = check_plan(plan, cfg_arm)
    for p in problems:
        print(f"  ✗ {p}")
    if args.dry_run:
        print("\n--dry-run：只打印计划，没连任何硬件。")
        return 0 if not problems else 1
    if problems:
        print("计划没通过检查，不跑。")
        return 1

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    (out / "top").mkdir(parents=True, exist_ok=True)
    (out / "wrist").mkdir(parents=True, exist_ok=True)
    (out / "side").mkdir(parents=True, exist_ok=True)
    print(f"\n输出 → {out}")
    if not args.mock:
        print("=" * 66)
        print("  ⚠ 臂会沿示范轨迹自己动（慢速）。开始前：**白板上什么都不要放**（套筒也拿走），")
        print("    臂停在起始位。会自己跑完；只有急停才按 q（走完当前一小步就停）。")
        print("=" * 66)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    rig = Rig(args, out)
    q = plan["q"]
    try:
        rig.connect()
        q_now = rig.joints()
        diff = np.degrees(np.abs(q_now - q[0])).max()
        if diff > 3.0 and not args.mock:
            raise RuntimeError(f"臂不在示范起点（最大差 {diff:.1f}°）。先 python home_arm.py 回位再跑")
        with Keys(mock=args.mock) as keys:
            print(f"手 → 开：{rig.set_hand('open')}")

            print("\nP1 走到抓取位姿（白板上没有套筒），顺路拍")
            for i in plan["shoot"][1:]:
                rig.move(q[i], keys)
                rig.shoot("P1_empty", i, plan)

            print("\n\n臂已在示范的抓取位姿，手张开。")
            ans = keys.ask("  把蓝套筒放进拇指和食指之间（摆成示范时那样），放好回车；输入 q 中止: ")
            if ans == "q":
                raise StopRequested("放套筒时选择中止")
            rig.sleeve = True

            print("\nP2 抓取位姿拍一张，然后倒着退回起点，逐点拍")
            g = plan["i_grasp"]
            rig.shoot("P2_at_grasp", g, plan)
            for i in reversed(plan["shoot"][:-1]):
                rig.move(q[i], keys)
                rig.shoot("P2_back", i, plan)

            print("\n\nP3 正着走回抓取位姿（不拍）")
            for i in plan["transit"][1:]:
                rig.move(q[i], keys)
                print(f"\r  示范 t={plan['t'][i]:5.1f}s", end="", flush=True)

            print("\n\nP4 合手")
            ang = rig.set_hand("close")
            # 食指实际下发的是耦合钳位后的值（拇指弯 400 时下限 500），按它比；
            # 示范里被套筒挡住时停在 ~652 / ~600，空夹则会走到指令值附近。
            # 10-13 真跑：两指捏套筒边沿时食指 565 / 拇指 739（指令 500/400）——食指只多 65，
            # 却是真夹住了（腕部画面里抬起后套筒还在两指间）。所以：两指都多出 40，或任一指多出 150。
            idx_cmd = max(HAND_CLOSE[0], rig.ih.index_floor(HAND_CLOSE[1]))
            ex_i, ex_t = ang[INDEX] - idx_cmd, ang[THUMB_BEND] - HAND_CLOSE[1]
            held = (ex_i > 40 and ex_t > 40) or max(ex_i, ex_t) > 150
            print(f"  实测 食指 {ang[INDEX]} / 拇指弯 {ang[THUMB_BEND]}"
                  f"（实际指令 {idx_cmd}/{HAND_CLOSE[1]}）→ {'被挡住 = 夹住了' if held else '⚠ 没被挡住'}")
            if not held and not args.mock:
                ans = keys.ask("  手指没被套筒挡住（可能没夹到）。仍然抬起输入 go，否则回车 = 张手中止: ")
                if ans != "go":
                    rig.set_hand("open")
                    raise StopRequested("没夹住，已张手")
            rig.shoot("P4_closed", g, plan)

            print("\nP5 抬起，逐点拍")
            for i in plan["lift"][1:]:
                rig.move(q[i], keys)
                rig.shoot("P5_lift", i, plan)

            print("\n\nP6 原路放回 → 张手 → 拍")
            for i in reversed(plan["lift"][:-1]):
                rig.move(q[i], keys)
            print(f"  手 → 开：{rig.set_hand('open')}")
            rig.shoot("P6_released", g, plan)

            print("\nP7 倒着退回起点（不拍）")
            for i in reversed(plan["transit"][:-1]):
                rig.move(q[i], keys)
                print(f"\r  示范 t={plan['t'][i]:5.1f}s", end="", flush=True)
            print("\n\n完成，臂回到示范起点。")
    except StopRequested as e:
        print(f"\n\n停下：{e}。臂停在原地，手保持 {rig.hand_state}。")
    except Exception as e:  # noqa: BLE001
        print(f"\n\n出错停下：{type(e).__name__}: {e}。臂停在原地，手保持 {rig.hand_state}。")
    finally:
        rig.save_meta(plan)
        print(f"已拍 {len(rig.rows)} 组 → {out}")
        rig.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default=str(DEFAULT_TRAJ))
    ap.add_argument("--t-grasp", type=float, default=DEFAULT_T_GRASP)
    ap.add_argument("--t-release", type=float, default=DEFAULT_T_RELEASE)
    ap.add_argument("--step-mm", type=float, default=15.0, help="拍照点间隔：末端走过这么多 mm")
    ap.add_argument("--step-deg", type=float, default=3.0, help="……或任一关节转过这么多度")
    ap.add_argument("--speed", type=float, default=100.0, help="MoveAbsJ 速度参数（驱动平时 200）")
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_replay")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="假臂 + 假手 + 假相机，验证流程")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
