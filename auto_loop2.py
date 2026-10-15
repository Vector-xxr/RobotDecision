#!/usr/bin/env python3
"""RobotDecision 去示教版全自动抓取（10-13 用户定的流程）。auto_loop.py（示教朝向版）保留不动。

第一阶段（起始朝向，不转）
  起始位竖直抬到 z340。每步先问腕部「套筒在哪」：看不到 → 只看俯视图选箭头走 20mm；
  看到但不在环里 → 俯视 + 腕部同字母箭头走 15mm。
进环后（10-13 用户：调位置和调姿态并行）每步问两道题
  第一题（腕部 + 侧面）：这一步调位置还是调姿态；空夹之后会附上「这个角度刚才没夹住」
  调姿态 → 第二题只看腕部：六种旋转（绕法兰 x/y/z 正反各 10°，转动中心在两指之间那一点），配参考图；
           累计偏离起始朝向超过 MAX_ROT_DEG 就不再问、只调位置
  调位置 → 第二题腕部 + 侧面：把套筒中心对到捏合点（箭头 5mm）、下降、合手、或「不在环里了」。下降每 2mm 读一次手指受力，
  任何一路偏离静止值 > TOUCH_TH 就是碰到了（板或套筒）→ 抬 2mm → 合手。
合手后看手指读数：夹住 → 抬 8cm（每步看手指）→ 放回 → 回起始位；
  空夹 → 张手、抬 3cm，回到「进环后先转」，告诉模型这个角度刚才没夹住（这一次不给「不用转」）。最多 MAX_ATTEMPTS 次。

不再用示教的任何东西：
  朝向     = 起始位朝向（+ 模型选的旋转）
  腕部瞄准点 = 手在空中合拢时拇指食指尖在腕部画面里的位置（touch_test 拍的 (311,263)；相机和手固连，和朝向无关）
  停止高度 = 指尖碰板（手指受力；touch_test：张开 950 时无名指 +3/mm、静止 ±2；1000 顶限位读不到）
  腕部箭头 = 起始朝向下实测的 J（calib_start 第一次真跑 +60 层，残差 1.5px），转了以后按绕竖直轴的分量旋转
安全（物理的）：逆解可达、回代、软限位、单步关节 ≤10°、法兰不低于 hard_min（随转角放宽）、到底没碰板就合手试；连续 3 步被拒就停；q 急停

    python auto_loop2.py --dry-run
    python auto_loop2.py --mock
    python auto_loop2.py            # 真跑（开始前输入 yes）；--auto 不等确认
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_capture as rc  # noqa: E402
import eval_decisions as E  # noqa: E402
import closed_loop as CL  # noqa: E402
import auto_loop as A  # noqa: E402

LIFT_Z = 340.0
STEP_FAR, STEP_MID, STEP_RING, STEP_DOWN, RAISE, RETREAT = 20.0, 15.0, 5.0, 12.0, 10.0, 50.0   # 降：每 2mm 查受力，碰到就停
LIFT_STEPS, MAX_ATTEMPTS, MAX_STEPS, MAX_ROT_DEG = 8, 4, 160, 45.0   # 偏离起始朝向超过 45° 就只调位置
ROT_DEG, SUB_MM, SUB_DEG = 10.0, 10.0, 2.0
TOUCH_TH, TOUCH_STEP = 8.0, 2.0
HARD_MIN_Z = 262.0               # 起始朝向人量的下限 291.9 再低 30mm：碰板检测失灵时的最后一道
MM_PER_DEG = 3.5                 # 手从法兰到指尖约 20cm：偏离起始朝向每 1°，指尖相对法兰最多升降约 3.5mm


def hard_min(R, R0):
    """法兰最低线随转角放宽。10-13 换位置那次：转了 17.8° 后法兰降到 263 指尖还离板约 4cm，
    固定的 262 把能抓的位置挡住了。指尖仍是全手最低的部位，碰板检测在前面把关。"""
    return HARD_MIN_Z - MM_PER_DEG * np.degrees(CL.axis_angle(R0.T @ R)[1])
OPEN_IDX_THUMB, OTHERS_OPEN = (880, 950), 950      # 1000 顶在机械限位上，碰板的力读不到
PINCH_PX, RING_R = (311, 263), 90
J0 = np.array([[1.5208, 1.0669], [1.3733, -1.5454]])   # 起始朝向、+60 层：套筒像素 = c0 + J0·(法兰水平偏移)
PIVOT0 = np.array([135.0, -127.0, -100.0])            # 起始朝向下两指之间那一点相对法兰（基座 mm）；水平是转腕实测，竖直是估的
REF = HERE / "data" / "robotdecision_axis_20261013_152719"   # 原件在 ~/atom_episodes/robotdecision_axis/，拷进项目免得迁移时丢
ROTS = {"x+": (0, 1), "x-": (0, -1), "y+": (1, 1), "y-": (1, -1), "z+": (2, 1), "z-": (2, -1)}
REF_TAGS = {"x+": "01_x+20", "x-": "02_x-2", "y+": "03_y+13", "y-": "04_y-20", "z+": "05_z+20", "z-": "06_z-20"}
DIR_BASE, LETTER_DIRS = CL.DIR_BASE, A.LETTER_DIRS


# ── 几何 ────────────────────────────────────────────────────────────────
def Rz2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def yaw_of(M):
    return float(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])))


def J_of(R, R0):
    """绕竖直轴转 ψ 时手-桌几何不变，像素对法兰的导数 = J0·Rz(−ψ)；俯仰/翻滚只取它绕竖直轴的分量近似。"""
    return J0 @ Rz2(-yaw_of(R @ R0.T))


def pinch_xy(p, R, R0):
    return (p + R @ R0.T @ PIVOT0)[:2]


# ── 画图 ────────────────────────────────────────────────────────────────
def ring(img):
    cv2.circle(img, PINCH_PX, RING_R, (0, 0, 0), 6)
    cv2.circle(img, PINCH_PX, RING_R, (0, 255, 0), 3)
    cv2.circle(img, PINCH_PX, 7, (0, 0, 0), -1)
    cv2.circle(img, PINCH_PX, 5, (0, 255, 0), -1)
    return img


def wrist_arrows(wrist, letters, J):
    img = ring(wrist.copy())
    h, w = img.shape[:2]
    o = np.asarray(PINCH_PX, float)
    for letter, d in letters.items():
        v = -(J @ np.array(DIR_BASE[d][:2], float))
        v /= np.linalg.norm(v)
        a = np.clip(o + v * (RING_R + 8), [10, 10], [w - 10, h - 10])
        b = np.clip(o + v * (RING_R + 70), [10, 10], [w - 10, h - 10])
        col = E.ARROW_BGR[letter]
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (0, 0, 0), 5, tipLength=0.3)
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), col, 3, tipLength=0.3)
        perp = np.array([v[1], -v[0]])
        c = np.clip(o + v * (RING_R + 40) + perp * 17, [11, 11], [w - 11, h - 11]).astype(int)
        cv2.circle(img, tuple(c), 10, (255, 255, 255), -1)
        cv2.circle(img, tuple(c), 10, col, 2)
        cv2.putText(img, letter, (int(c[0]) - 6, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


def top_arrows(top, p, R, R0, letters):
    img = top.copy()
    E.draw_arrows(img, E.TOP_MAP(pinch_xy(p, R, R0) - PIVOT0[:2]), letters)   # 起点 = 两指之间那一点
    return img


_REF_CACHE = {}


def ref_sheet(rot_letters, allowed):
    """参考图：NOW = 不转时腕部看到的，A~F = 各种转完之后（10-13 axis_look 真机拍，z400、套筒在画面里）。"""
    def tile(tag):
        if tag not in _REF_CACHE:
            im = cv2.imread(str(REF / f"{tag}_cam_front.jpg"))
            _REF_CACHE[tag] = cv2.resize(im if im is not None else np.zeros((480, 640, 3), np.uint8), (320, 240))
        return _REF_CACHE[tag].copy()
    tiles = []
    for lab, tag in [("NOW", "00_level")] + [(k, REF_TAGS[v]) for k, v in rot_letters.items() if v in allowed]:
        t = tile(tag)
        cv2.rectangle(t, (0, 0), (90 if lab == "NOW" else 40, 34), (255, 255, 255), -1)
        cv2.putText(t, lab, (5, 26), 0, 0.9, (0, 0, 0), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


# ── 题目 ────────────────────────────────────────────────────────────────
ROT_TEXT = ("Image 1 is from a camera fixed on a robot hand, looking down past the hand at a table. The dark finger at "
            "the LEFT edge is the INDEX finger and the dark finger at the RIGHT edge is the THUMB. The hand grasps by "
            "pinching an object between the thumb tip and the index fingertip; the small green dot inside the green "
            "ring marks where the two fingertips meet when the hand closes. Before going down to grasp the blue "
            "sleeve, the hand can be rotated. Image 2 is a reference sheet from the same camera: panel NOW is the "
            "view in the hand's normal orientation, and each lettered panel shows the view after that rotation "
            "(photographed earlier, higher above the table).")
ROT_Q = ("Should the hand be rotated so that the thumb and the index finger end up on the two opposite sides of the "
         "blue sleeve, ready to pinch it? If they are already placed well for pinching, choose keep.")
ROT_FAIL = (" IMPORTANT: a grasp attempt in the current orientation just failed - the hand went down and closed, but "
            "the thumb and index finger closed on nothing. Choose a rotation; keeping the orientation is not allowed.")


def ask_rot(dec, wrist, rot_letters, allowed, failed, rejected, allow_keep=True):
    w = ring(wrist.copy())
    sheet = ref_sheet(rot_letters, allowed)
    ch = [(f"rot_{k}", f"Rotation {k} (panel {k} in image 2).") for k, v in rot_letters.items() if v in allowed]
    if not failed and allow_keep:
        ch.append(("keep", "No rotation: the thumb and index finger are already placed well for pinching."))
    text = ROT_TEXT + (ROT_FAIL if failed else "")
    if rejected:
        text += " (Some rotations were left out because the arm cannot turn further that way.)"
    req = {"model": E.MODEL, "input": A.msg(text, ("Image 1 (hand camera, now):", w), ("Image 2 (reference):", sheet)),
           "questions": A.two_orders("rot", ROT_Q, ch)}
    return dec(req, "rot"), w, sheet


WHAT_TEXT = ("Image 1 is from a camera fixed on a robot hand, looking down past the hand. The dark finger at the LEFT "
             "edge is the INDEX finger and the dark finger at the RIGHT edge is the THUMB; the green ring with a dot marks "
             "where the two fingertips meet when the hand closes. Image 2 is from a camera at table level that shows the "
             "fingertips and the blue sleeve from the side. The hand must pinch the blue sleeve between the thumb tip and "
             "the index fingertip.")
WHAT_Q = ("What should be adjusted next? Choose position to move the hand sideways, lower it or close the fingers; choose "
          "orientation to rotate the hand so that the thumb and the index finger sit better on the two opposite sides of "
          "the sleeve.")


def ask_what(dec, wrist, side, failed):
    """10-13 用户：进环后调位置和调姿态并行 —— 每步先问调哪个（腕部+侧面），再问具体怎么调。"""
    w = ring(wrist.copy())
    ch = [("position", "Adjust the hand's position: move it sideways, lower it, or close the fingers."),
          ("orientation", "Rotate the hand: the thumb and index finger are not placed well around the sleeve.")]
    text = WHAT_TEXT + (ROT_FAIL.replace(" Choose a rotation; keeping the orientation is not allowed now.", "")
                        if failed else "")
    req = {"model": E.MODEL, "input": A.msg(text, ("Image 1 (hand camera):", w), ("Image 2 (side camera):", side)),
           "questions": A.two_orders("w", WHAT_Q, ch)}
    return dec(req, "what"), w


def ask_phase(dec, wrist):
    img = ring(wrist.copy())
    req = {"model": E.MODEL, "input": A.msg(A.PHASE_TEXT, ("", img)), "questions": A.two_orders("p", A.PHASE_Q, A.PHASE_CH)}
    return dec(req, "phase"), img


FAR_Q2 = ("Which arrow moves the robot hand toward the blue sleeve? If the hand (where the arrows start) is already "
          "right above the blue sleeve or within a few centimetres of it, choose over_sleeve.")


def ask_far(dec, top, p, R, R0, letters):
    """10-13 第一次真跑：只凭腕部「看到了」就切到进环前，腕部被垫子上的橙色方框骗，手在白板外转了 100 多步。
    改成由俯视图自己说「手已经在套筒上方」才往下一段走。"""
    img = top_arrows(top, p, R, R0, letters)
    ch = [(f"arrow_{k}", f"Arrow {k}.") for k in "ABCD"] + [
        ("over_sleeve", "The hand is already right above the blue sleeve or within a few centimetres of it."),
        ("target_not_visible", "No blue sleeve is visible.")]
    req = {"model": E.MODEL, "input": A.msg(A.FAR_TEXT, ("", img)), "questions": A.two_orders("f", FAR_Q2, ch)}
    return dec(req, "far"), img


MID_TEXT2 = ("Image 1 is from a fixed camera looking straight down at a work table; the robot arm reaches in from the "
             "bottom-left. Image 2 is from a camera fixed on the robot hand, looking down; the green ring marks where an "
             "object must be for the fingers to grasp it. The same four lettered arrows are drawn in both images, and "
             "arrow X is the same hand move in both images (they point differently because the hand camera is turned). "
             "Task: move the hand toward the blue sleeve until the sleeve is inside the green ring.")
MID_Q2 = ("Choose the arrow that points from the green ring toward the blue sleeve in image 2. (In image 1 this is the "
          "arrow that points from the hand toward the sleeve.)")


def ask_mid(dec, top, wrist, p, R, R0, letters, J):
    """10-13 第二次真跑：旧问法「选哪个箭头让套筒朝绿环靠近」有歧义（手往东走、画面里套筒往西移），
    正倒序东西反着答，手来回摆 40 步。线下 10 张真帧：旧问法 8/20，改成「从绿环指向套筒」20/20 且两序全一致。"""
    t = top_arrows(top, p, R, R0, letters)
    w = wrist_arrows(wrist, letters, J)
    ch = [(f"arrow_{k}", f"Arrow {k}.") for k in "ABCD"]
    req = {"model": E.MODEL, "input": A.msg(MID_TEXT2, ("Image 1 (top camera):", t), ("Image 2 (hand camera):", w)),
           "questions": A.two_orders("m", MID_Q2, ch)}
    return dec(req, "mid"), t, w


def ask_ring(dec, wrist, side, letters, J):
    w = wrist_arrows(wrist, letters, J)
    ch = [(f"arrow_{k}", f"Move along arrow {k}.") for k in "ABCD"]
    ch += [("lower", "Lower the hand a short step."), ("close", "Close the thumb and index finger to grasp."),
           ("outside", "The blue sleeve is no longer inside the green ring in image 1.")]
    req = {"model": E.MODEL,
           "input": A.msg(A.RING_TEXT, ("Image 1 (hand camera):", w), ("Image 2 (side camera):", side)),
           "questions": A.two_orders("r", A.RING_Q, ch)}
    return dec(req, "ring"), w


class ApiDown(Exception):
    pass


def make_decider(args, out):
    if not args.mock:
        base = CL.make_decider(args, out)

        def robust(req, kind):
            """10-13 换位置重跑到第 55 步：决策接口回 503「Decision inference is unavailable」，原来直接抛出去、
            臂停在套筒旁边的低处。现在服务端/连接/超时错误退避重试约 2 分钟，还不行就 ApiDown → 走正常收尾回起始位。"""
            from openai import APIConnectionError, APITimeoutError, InternalServerError
            for w in (3, 6, 12, 24, 30, 30, None):
                try:
                    return base(req, kind)
                except (InternalServerError, APIConnectionError, APITimeoutError) as e:
                    if w is None:
                        raise ApiDown(f"{type(e).__name__}: {str(e)[:120]}") from e
                    print(f"  决策接口暂时不可用（{type(e).__name__}），{w}s 后重试 ...", flush=True)
                    time.sleep(w)
        return robust
    rnd = random.Random(2)
    script = {"phase": ["outside", "outside"] + ["inside"] * 80,
              "far": ["arrow_A", "arrow_B", "over_sleeve"], "mid": ["arrow_C", "arrow_A"],
              "rot": ["rot_A", "rot_B", "rot_C", "rot_D", "rot_E"],
              "what": ["orientation", "position", "position", "position", "position", "position", "position",
                       "orientation", "position"],
              "ring": ["arrow_B"] + ["lower"] * 60}
    pos = {k: 0 for k in script}

    def fake(req, kind):
        seq = script[kind]
        v = seq[min(pos[kind], len(seq) - 1)]
        pos[kind] += 1
        values = [c["value"] for c in req["questions"][0]["choices"]]
        if v not in values:                                   # 剧本里的字母这次不在选项里（被拿掉了）
            v = values[0]
        b = v if rnd.random() > 0.25 else values[-1]
        return {"answers": {"a": {"choice": v}, "b": {"choice": b}}, "ms": 1.0, "input_tokens": 0}
    return fake


# ── 运动 ────────────────────────────────────────────────────────────────
class Arm:
    """末端位姿（法兰 p mm、朝向 R）直线小步规划：每小步逆解检查，全部可行才执行。"""

    def __init__(self, rig, kin, keys):
        self.rig, self.kin, self.keys = rig, kin, keys

    def now(self):
        q = self.rig.joints()
        T = self.kin.fk(q)
        return q, T[:3, 3] * 1000, T[:3, :3]

    def plan(self, path, floor=HARD_MIN_Z, start=None):
        """path：[(p, R), ...] 关键位姿；相邻之间按 ≤10mm、≤2° 细分。返回 (关节序列, 原因)。
        floor：法兰最低 z；只有起步和回起始位那段竖直路径（在白板侧面）才放开。
        start：从哪个位姿起算（默认实测）。10-13 第一次真跑：每步从实测位姿起算，高度每步漂 +0.3mm、
        朝向每步漂 0.05°，100 步后高了 33mm、歪了 4.4° —— 所以主循环传上一步的指令位姿，实测关节只当逆解起点。"""
        q, p, R = self.now()
        if start is not None:
            p, R = np.asarray(start[0], float), start[1]
        seq = []
        for p1, R1 in path:
            p1 = np.asarray(p1, float)
            ax, ang = CL.axis_angle(R.T @ R1)
            n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p) / SUB_MM, np.degrees(ang) / SUB_DEG))))
            for nn in (n, 4 * n):             # 单步关节转太多：接近奇异时切细就能过；真要跳支切细也过不去
                part, qq, why = [], q, ""
                for k in range(1, nn + 1):
                    f = k / nn
                    qk, why = self.kin.solve(qq, p + (p1 - p) * f, R @ CL.rot(ax, ang * f), floor)
                    if qk is None:
                        break
                    part.append(qk)
                    qq = qk
                else:
                    break
                if "一步关节要转" not in why:
                    return None, why
            else:
                return None, why
            seq += part
            q, p, R = qq, p1, R1
        return seq, ""

    def run(self, seq):
        for q in seq:
            self.rig.move(q, self.keys)

    def go(self, path, floor=HARD_MIN_Z, start=None):
        seq, why = self.plan(path, floor, start)
        if seq is None:
            return False, why
        self.run(seq)
        return True, ""


def forces(hand, n=3):
    return np.median(np.array([hand.get_force() for _ in range(n)], float), axis=0)


# ── 主流程 ───────────────────────────────────────────────────────────────
def run(args):
    from atom_robot.ar5.config import AR5Config
    kin = CL.Kin()
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    T0 = kin.fk(q_home)
    R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
    # 离线演算：起步 + 在白板中间各转 10° 能不能走
    lift_path = [(np.array([p_home[0], p_home[1], z]), R0)
                 for z in list(np.arange(p_home[2] + 10, LIFT_Z - 0.01, 10)) + [LIFT_Z]]
    print(f"起步：起始位竖直抬到 z{LIFT_Z:.0f}，之后全程起始朝向（工具 z {R0[:, 2].round(2)}）+ 模型选的旋转")
    print(f"腕部瞄准点 {PINCH_PX}、环半径 {RING_R}；碰板门槛 {TOUCH_TH:g}（每 {TOUCH_STEP:g}mm 读一次）；"
          f"法兰最低 {HARD_MIN_Z:.0f}（每偏离起始朝向 1° 放宽 {MM_PER_DEG:g}mm）；旋转每次 {ROT_DEG:.0f}°、累计不超过 {MAX_ROT_DEG:.0f}°；合手最多试 {MAX_ATTEMPTS} 次")
    missing = [t for t in ["00_level"] + list(REF_TAGS.values()) if not (REF / f"{t}_cam_front.jpg").exists()]
    if missing:
        print(f"✗ 参考图缺：{missing}")
        return 1
    if args.dry_run:
        print("离线演算：只检查参数和参考图，没连硬件、没调接口。")
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"记录 → {out}")
    if not args.mock and not args.auto:
        print("=" * 70)
        print("  ⚠ 全自动、不用示教：臂抬起后由模型找套筒、进环后先决定怎么转、再对准下降，碰板就停、合手。")
        print("    开始前：套筒立在白板上，臂在起始位。随时按 q 急停。")
        print("=" * 70)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    rc.HAND_OPEN, rc.OTHERS_OPEN = OPEN_IDX_THUMB, OTHERS_OPEN
    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    dec = make_decider(args, out)
    rng = np.random.default_rng(int(time.time()))
    log, t0 = [], time.time()
    what_pool = []
    st = {"closed": False, "lifted": 0, "attempts": 0, "rot_done": False, "rot_n": 0, "failed": False,
          "rejected": 0, "no_rot": set(), "stage": "far", "nv": 0}
    phase_pool, act_pool, last_kind = [], [], None
    step, result = 0, "未完成"
    try:
        rig.connect()
        if np.degrees(np.abs(rig.joints() - q_home)).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在起始位。先回位（灵巧手遥操/VR设备/home_arm.py）再跑")
        with rc.Keys(mock=args.mock) as keys:
            arm = Arm(rig, kin, keys)
            rig.set_hand("open")
            print("\n起步：竖直抬起 ...", flush=True)
            lift_seq, why = arm.plan(lift_path, floor=-1e9)
            if lift_seq is None:
                raise RuntimeError(f"起步走不了：{why}")
            arm.run(lift_seq)
            cmd = [lift_path[-1][0].copy(), R0.copy()]          # 指令位姿：规划都从它起算，实测关节只当逆解起点

            def go(path, floor=None):
                if floor is None:
                    floor = hard_min(path[-1][1], R0)
                ok, why = arm.go(path, floor, start=cmd)
                if ok:
                    cmd[0], cmd[1] = np.asarray(path[-1][0], float), path[-1][1]
                return ok, why

            def touched_descend(dz):
                """往下 dz mm，每 TOUCH_STEP 读一次受力。返回 (是否碰到, 说明)。"""
                base = forces(rig.hand, 7)
                p, R = cmd
                target = p[2] - dz
                while p[2] - TOUCH_STEP >= target - 1e-6:
                    nxt = p + [0, 0, -TOUCH_STEP]
                    ok, why = go([(nxt, R)])
                    if not ok:
                        # 10-13 换位置那次：降不下去没算被拒，模型一直选「下降」原地空转 60 多步。现在到底就合手试
                        return True, f"法兰 {p[2]:.1f} 降不下去（{why}），没测到碰板，就在这里合手试"
                    p = nxt
                    f = forces(rig.hand)
                    if args.mock and p[2] <= 285:
                        f = base + [0, 20, 0, 0, 0, 0]
                    d = f - base
                    k = int(np.argmax(np.abs(d)))
                    if abs(d[k]) > TOUCH_TH:
                        go([(p + [0, 0, TOUCH_STEP], R)])              # 抬 2mm 卸掉压力再合
                        return True, f"法兰 {p[2]:.1f} 碰到（{['小指', '无名指', '中指', '食指', '拇指弯', '拇指转'][k]} {d[k]:+.0f}）"
                return False, f"降了 {dz:.0f}mm 没碰到"

            def try_close(rec):
                st["attempts"] += 1
                ang = rig.set_hand("close")
                held, ex_i, ex_t = CL.finger_held(ang)
                if args.mock:
                    held = st["attempts"] >= 2
                rec["grasp"] = {"try": st["attempts"], "index": int(ang[rc.INDEX]), "thumb": int(ang[rc.THUMB_BEND]),
                                "held": bool(held)}
                print(f"  合手第 {st['attempts']} 次：食指 {ang[rc.INDEX]} / 拇指 {ang[rc.THUMB_BEND]}"
                      f"（+{ex_i}/+{ex_t}）→ {'夹住了' if held else '空夹'}", flush=True)
                if held:
                    st["closed"] = True
                    return
                rig.set_hand("open")
                p, R = cmd
                go([(p + [0, 0, RETREAT], R)])
                st.update(rot_done=False, rot_n=0, failed=True)
                print(f"  空夹：张手、抬 {RETREAT:.0f}mm，回到环内并告诉模型这个角度刚才没夹住", flush=True)

            try:
                while step < MAX_STEPS:
                    step += 1
                    keys.poll()
                    q, p_meas, _ = arm.now()
                    p, R = cmd[0].copy(), cmd[1]
                    J = J_of(R, R0)
                    fr = CL.grab(rig)
                    top, wrist, side = fr.get("cam_high"), fr.get("cam_front"), fr.get("cam_side")
                    for k, im in (("top", top), ("wrist", wrist), ("side", side)):
                        if im is not None:
                            cv2.imwrite(str(out / f"{step:03d}_{k}.jpg"), im)
                    rot_from_R0 = np.degrees(CL.axis_angle(R0.T @ R)[1])
                    rec = {"step": step, "t": round(time.time() - t0, 1), "flange": p.round(1).tolist(),
                           "flange_meas": p_meas.round(1).tolist(), "stage_state": st["stage"],
                           "q_deg": np.degrees(q).round(2).tolist(),
                           "rot_from_start_deg": round(rot_from_R0, 1)}
                    panels = [top, wrist] + ([side] if side is not None else [])
                    action, path, note = None, None, ""

                    if st["closed"]:
                        ang = rig.hand.get_angles()
                        held = CL.finger_held(ang)[0] or args.mock
                        rec["fingers"] = [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])]
                        if not held:
                            action, note = "stop", "抬起途中手指不再被挡住 = 掉了"
                        elif st["lifted"] >= LIFT_STEPS:
                            action, note = "done", f"已抬 {st['lifted'] * RAISE:.0f}mm，手指仍被挡住"
                        else:
                            action, path, note = "raise", [(p + [0, 0, RAISE], R)], "夹着，按规则抬"
                    else:
                        if st["stage"] == "far":
                            phase = "far"
                        elif st["stage"] == "ring":
                            # 10-13 第四次真跑：第二阶段 83 步、每步先问位置题再问对准题，调用翻倍。
                            # 进环后不再问位置题；对准题里有「不在环里了」出口
                            phase = "inside"
                        else:
                            res, w_img = ask_phase(dec, wrist)
                            pa, pb = res["answers"]["a"]["choice"], res["answers"]["b"]["choice"]
                            rec["phase"] = [pa, pb]
                            phase = A.vote(phase_pool, pa, pb)
                            panels[1] = w_img
                            if phase == "not_visible":
                                st["nv"] += 1
                                phase = "far" if st["nv"] >= 2 else None
                                if phase == "far":
                                    st["stage"] = "far"
                            elif phase is not None:
                                st["nv"] = 0
                                if phase == "inside":
                                    st["stage"] = "ring"
                        if phase is None:
                            action, note = "reask", "位置题没定下来（两次不一致，或腕部第一次说看不到）"
                        else:
                            letters = dict(zip("ABCD", rng.permutation(LETTER_DIRS)))
                            what = None
                            if phase == "inside":
                                allowed = [r for r in ROTS if r not in st["no_rot"]]
                                if rot_from_R0 >= MAX_ROT_DEG or not allowed:
                                    what = "position"                     # 转够了 / 哪都转不动了：只调位置
                                else:
                                    res_w, _ = ask_what(dec, wrist, side, st["failed"])
                                    wa, wb = res_w["answers"]["a"]["choice"], res_w["answers"]["b"]["choice"]
                                    rec["what"] = [wa, wb]
                                    what = A.vote(what_pool, wa, wb)
                            if phase == "inside" and what is None:
                                kind = "what"
                                res = {"answers": {"a": {"choice": "reask"}, "b": {"choice": "reask"}}}
                                sem = lambda x: x  # noqa: E731
                                stage = "环内(先问调位置还是调姿态)"
                            elif phase == "inside" and what == "orientation":
                                kind = "rot"
                                rot_letters = dict(zip("ABCDEF", rng.permutation(list(ROTS))))
                                res, w_img, sheet = ask_rot(dec, wrist, rot_letters, allowed, st["failed"], st["no_rot"],
                                                            allow_keep=False)
                                panels[1], panels[0] = w_img, sheet
                                sem = lambda x: rot_letters[x[-1]] if x.startswith("rot_") else x  # noqa: E731
                                stage = "环内·调姿态(只看腕部)"
                            elif phase == "inside":
                                kind = "ring"
                                res, panels[1] = ask_ring(dec, wrist, side, letters, J)
                                sem = lambda x: letters[x[-1]] if x.startswith("arrow_") else x  # noqa: E731
                                stage = "环内·调位置(腕部+侧面)"
                            elif phase in ("outside", "inside"):          # 进环前（inside 已在上面处理）
                                kind = "mid"
                                res, panels[0], panels[1] = ask_mid(dec, top, wrist, p, R, R0, letters, J)
                                sem = lambda x: letters[x[-1]]  # noqa: E731
                                stage = "进环前(俯视+腕部)"
                            else:
                                kind = "far"
                                res, panels[0] = ask_far(dec, top, p, R, R0, letters)
                                sem = lambda x: letters[x[-1]] if x.startswith("arrow_") else x  # noqa: E731
                                stage = "粗定位(只看俯视)"
                            if kind != last_kind:
                                act_pool.clear()
                            last_kind = kind
                            a, b = sem(res["answers"]["a"]["choice"]), sem(res["answers"]["b"]["choice"])
                            rec.update(stage=stage, answer=[a, b])
                            if a != b and a in DIR_BASE and b in DIR_BASE and np.dot(DIR_BASE[a], DIR_BASE[b]) == 0:
                                choice = (a, b)          # 第四次真跑重问 45 次，多是「东/南」这种相邻方向：两个都对，走斜对角
                            else:
                                choice = A.vote(act_pool, a, b) if kind != "what" else None
                            if kind == "what":
                                action, note = "reask", f"{stage}：两次不一致 {rec['what'][0]}/{rec['what'][1]}"
                            elif choice is None:
                                action, note = "reask", f"{stage}：两次不一致 {a}/{b}"
                            else:
                                note = (f"{A.MAX_REASK} 轮不一致，多数票；" if a != b and not isinstance(choice, tuple) else "") + stage + "："
                                step_mm = {"far": STEP_FAR, "mid": STEP_MID, "ring": STEP_RING}.get(kind, 0)
                                if isinstance(choice, tuple):
                                    v = (np.array(DIR_BASE[choice[0]], float) + DIR_BASE[choice[1]]) / np.sqrt(2)
                                    action = f"move_{choice[0]}+{choice[1]}"
                                    path = [(p + v * step_mm, R)]
                                    note += f"{choice[0]}/{choice[1]} 相邻，走斜对角 {step_mm:.0f}mm"
                                elif choice in DIR_BASE:
                                    action = f"move_{choice}"
                                    path = [(p + np.array(DIR_BASE[choice]) * step_mm, R)]
                                    note += f"{choice} {step_mm:.0f}mm"
                                elif choice in ROTS:
                                    ax, sg = ROTS[choice]
                                    R1 = R @ CL.rot(np.eye(3)[ax], np.radians(sg * ROT_DEG))
                                    c = p + R @ R0.T @ PIVOT0                    # 两指之间那一点不动
                                    path = [(c + R1 @ R.T @ (p - c), R1)]
                                    action, note = f"rotate_{choice}", note + f"绕末端 {choice[0]} 轴转 {sg * ROT_DEG:+.0f}°"
                                elif choice == "over_sleeve":
                                    st["stage"], st["nv"] = "near", 0
                                    action, note = "to_near", note + "俯视说手已在套筒上方 → 改看腕部"
                                elif choice == "outside":
                                    st["stage"] = "near"
                                    action, note = "to_mid", note + "套筒不在环里了 → 回进环前"
                                elif choice == "keep":
                                    action, note = "keep", note + "不用转，进第二阶段"
                                elif choice == "lower":
                                    action, note = "lower", note + f"下降 {STEP_DOWN:.0f}mm（碰到就停）"
                                elif choice == "close":
                                    action, note = "close", note + "合手"
                                elif choice == "target_not_visible":
                                    action, note = "stop", note + "俯视图里也看不到套筒"

                    # 执行
                    if path is not None:
                        seq, why = arm.plan(path, floor=hard_min(path[-1][1], R0), start=cmd)
                        if seq is None and (action.startswith("move_") or action == "raise"):
                            # 10-13 第三次真跑：套筒离底座近，法兰缩到 (304,-317) 再往东 5mm 就逆解不可达。
                            # 绕穿过两指之间那点的竖直轴转腕：手-桌几何不变（套筒在腕部画面里不动、箭头按转角换算），
                            # 只是各关节换个姿态，常能让开可达极限 → 先这样转，再走原来那一步
                            c = p + R @ R0.T @ PIVOT0
                            for yaw in (15, -15, 30, -30, 45, -45):
                                Rz = CL.rot(np.array([0, 0, 1.0]), np.radians(yaw))
                                p_y, R_y = c + Rz @ (p - c), Rz @ R
                                p_t = p_y + (np.asarray(path[-1][0], float) - p)
                                seq2, _ = arm.plan([(p_y, R_y), (p_t, R_y)], floor=hard_min(R_y, R0), start=cmd)
                                if seq2 is not None:
                                    seq, path = seq2, [(p_y, R_y), (p_t, R_y)]
                                    note += f"；原方向够不着，先绕竖直轴转 {yaw:+d}° 让开再走"
                                    break
                        if seq is None:
                            if action.startswith("rotate_"):
                                st["no_rot"].add(action.split("_")[1])          # 这个方向转不动了，以后不给
                            note += f"；✗ 不能执行：{why}"
                            action += "(被拒)"
                    rec.update(action=action, note=note)
                    print(f"step {step:3d}  t={rec['t']:5.1f}s  法兰 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) 转了 {rot_from_R0:4.1f}°"
                          f"  -> {action}  | {note}", flush=True)
                    panel = CL.show([x for x in panels if x is not None], f"step {step} -> {action}\n{note[:95]}",
                                    display=not args.mock and not args.auto)
                    cv2.imwrite(str(out / f"{step:03d}_panel.jpg"), panel)
                    log.append(rec)

                    if action in ("stop", "done"):
                        result = "完成" if action == "done" else f"停止：{note}"
                        break
                    if action.endswith("(被拒)"):
                        st["rejected"] += 1
                        if st["rejected"] >= 3:
                            result = f"停止：连续 3 步都走不了（{note.split('；✗ 不能执行：')[-1]}）"
                            break
                        continue
                    if action != "reask":
                        st["rejected"] = 0
                        phase_pool.clear(), act_pool.clear(), what_pool.clear()
                    if action == "keep":
                        st["rot_done"], st["failed"] = True, False
                    elif action.startswith("rotate_"):
                        arm.run(seq)
                        cmd[0], cmd[1] = np.asarray(path[-1][0], float), path[-1][1]
                        st["failed"] = False
                    elif action == "lower":
                        hit, msg_ = touched_descend(STEP_DOWN)
                        rec["descend"] = msg_
                        print(f"  {msg_}", flush=True)
                        if hit:
                            try_close(rec)
                    elif action == "close":
                        try_close(rec)
                    elif seq is not None and path is not None:
                        arm.run(seq)
                        cmd[0], cmd[1] = np.asarray(path[-1][0], float), path[-1][1]
                        if action == "raise":
                            st["lifted"] += 1
                    if not st["closed"] and st["attempts"] >= MAX_ATTEMPTS:
                        result = f"停止：合手 {st['attempts']} 次都空夹"
                        break
                else:
                    result = f"停止：到了 {MAX_STEPS} 步上限"

            except ApiDown as e:
                result = f"停止：决策接口不可用（{e}）"
                print(f"  {result}", flush=True)
            print(f"\n结果：{result}（{step} 步，{time.time() - t0:.0f}s，合手 {st['attempts']} 次）", flush=True)
            # 收尾：夹着就放回、张手；抬到 z340 → 转回起始朝向（绕两指之间那一点）→ 回起始位上方 → 降到起始位
            if st["closed"]:
                p, R = cmd
                go([(p - [0, 0, RAISE * st["lifted"]], R)])
            rig.set_hand("open")
            print("回起始位 ...", flush=True)
            # 10-13 两次真跑都在白板边缘「转回起始朝向」逆解不可达 → 先保持朝向平移回白板中间上方再转（rescue_home 同款，真机用过）
            # 10-13 并行方案第一次：放回套筒后手在底座附近 (234,-287,267)，竖直抬不到 360（夹着时也只到 346）→
            # 抬升高度从高往低试，取整条「抬起 → 平移回白板中间上方 → 转回起始朝向」都可行的第一个
            p, R = cmd
            back, why = None, ""
            for z in sorted({max(p[2], LIFT_Z + 20), 345.0, 330.0, 315.0, round(p[2] + 30, 1), round(float(p[2]), 1)}, reverse=True):
                if z < p[2] + (20 if p[2] < 320 else 0):     # 10-14：已在 340 高处时 360/370 够不着，允许就在当前高度出发
                    continue
                zc = max(z, LIFT_Z)
                path_back = [(np.array([p[0], p[1], z]), R), (np.array([450.0, 0.0, zc]), R), (np.array([450.0, 0.0, zc]), R0)]
                back, why = arm.plan(path_back, floor=-1e9, start=cmd)
                if back is not None:
                    break
            if back is None:
                raise RuntimeError(f"回到白板中间上方再转回起始朝向走不了：{why}。停在这里，用 rescue_home.py 回位")
            arm.run(back)
            cmd[0], cmd[1] = path_back[-1][0], R0
            ok, why = go([(np.array([p_home[0], p_home[1], LIFT_Z]), R0)])
            if not ok:
                raise RuntimeError(f"回起始位上方走不了：{why}")
            # 7 关节有冗余：转过腕再回来，同一末端位姿的关节角会差十几度（模拟里差 17°）。
            # 在 z340 高处分小步把关节过渡到起步时最高点那组，每步查法兰不降太多，再沿起步路径原路降回起始位
            q_now, q_top = rig.joints(), lift_seq[-1]
            n = max(1, int(np.ceil(np.degrees(np.abs(q_top - q_now)).max() / 3.0)))
            for k in range(1, n + 1):
                qk = q_now + (q_top - q_now) * k / n
                zk = kin.fk(qk)[2, 3] * 1000
                if zk < LIFT_Z - 40:
                    raise RuntimeError(f"关节过渡途中法兰会降到 {zk:.0f}mm，停在这里请手动回位")
                rig.move(qk, keys)
            for q in reversed(lift_seq[:-1]):
                rig.move(q, keys)
            rig.move(q_home, keys)
            print("已回到起始位。", flush=True)
    except rc.StopRequested as e:
        result = f"急停：{e}"
        print(f"\n{result}。臂停在原地，手 {rig.hand_state}。")
    except Exception as e:  # noqa: BLE001
        result = f"出错：{type(e).__name__}: {e}"
        print(f"\n{result}。臂停在原地，手 {rig.hand_state}。")
    finally:
        summary = {"result": result, "steps": step, "seconds": round(time.time() - t0, 1), "attempts": st["attempts"],
                   "log": log}
        (out / "log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"记录 → {out}")
        cv2.destroyAllWindows()
        rig.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_auto2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--auto", action="store_true", help="不等人输入 yes（没有终端时用）")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
