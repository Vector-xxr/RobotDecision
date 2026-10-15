#!/usr/bin/env python3
"""起始朝向下的一次性标定（为了去掉示教）：法兰下限、腕部箭头方向、环、转腕转轴。

量的都是机器人自己的东西（腕部相机装得多歪、指尖在哪、转腕绕哪一点），套筒只是参照物；
抓取运行时不检测套筒，模型照旧只看图选项。手全程保持起始位的朝向（离竖直约 42°、朝前），只在第 4、6 步绕竖直轴转。

为什么只标一个朝向就够：手绕竖直轴转 ψ 时，手和桌面的相对几何不变（桌面水平、套筒是竖直圆柱），所以
  环的像素位置、法兰下限都不变；腕部画面里的移动方向 = J·Rz(−ψ)，按角度算出来即可。

流程（每一小步都用自己的 PoE 逆解并检查：可达、回代 <1mm、软限位、单步关节 ≤10°；MoveAbsJ、实测到位才继续）
  1 下限    起始位竖直抬到 z340 → 平移到 --spot 上方（白板上先什么都不放）。人看着一次次往下降，
            最低那根指尖离板 3~5mm 时输入 ok → 这个法兰高度就是下限
  2 放套筒  抬到 下限+90，把套筒立在拇指和食指指尖中间的正下方
  3 网格    下限+90、+60 两层 5×5（±20mm，间距 10mm）只拍照 → 套筒像素 = c0 + J·(手的水平偏移)
  4 转腕    +90 层原地绕竖直轴转 −15°、−30° 拍照 → 解出套筒相对法兰的水平位置
  5 合手    3×3（±10mm）：每次先回参考点拍一张重新定位套筒（合手会把它带偏），再对准、降到下限、拍、
            合手、看手指读数、张手、抬回 → 夹得住的那些 = 环；它们的中心 = 抓取点（转腕就绕这一点转）
  6 验证    手对准套筒后绕抓取点转 −15°、−30°，套筒在腕部画面里应该几乎不动
  7 回起始位
结果写到 RobotDecision/calib_start.json（auto_loop 之后读它），原始图在 ~/atom_episodes/robotdecision_calib/

    python calib_start.py --dry-run
    python calib_start.py --mock
    python calib_start.py
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

LIFT_Z = 340.0
LAYERS = (90.0, 60.0)            # mm，比下限高多少；先 +90（离套筒顶远）再 +60
SPAN, STEP = 20.0, 10.0          # 网格 ±20mm、间距 10mm → 5×5
YAWS = (-15.0, -30.0)            # 转腕拍照的角度（往右转；离线算过 0~−60° 不换支）
TRIAL = 10.0                     # 合手试的偏移 ±10mm → 3×3
MOVE_STEP, YAW_STEP = 10.0, 5.0  # 每一小步最多走 10mm、转 5°
HARD_MIN_Z = 240.0               # 量下限时法兰绝对不低于这个（人没喊停时的兜底）
CALIB_FILE = HERE / "calib_start.json"


def Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Rz2(deg):
    return Rz(deg)[:2, :2]


def blob(img):
    """腕部图里最大的蓝色块：(中心 u, v, 等效半径 px, 面积)；外轮廓填实（套筒中间的孔不算缺口）。"""
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 800:
        return None
    mo = cv2.moments(c)
    return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], float(np.sqrt(a / np.pi)), float(a)


def grid_offsets():
    """蛇形：相邻两点只差一格。"""
    vals = np.arange(-SPAN, SPAN + 0.1, STEP)
    pts = []
    for j, dy in enumerate(vals):
        xs = vals if j % 2 == 0 else vals[::-1]
        pts += [(dx, dy) for dx in xs]
    return pts


def trial_offsets():
    """中心先试（人看着），再绕一圈。"""
    t = TRIAL
    return [(0, 0), (t, 0), (t, t), (0, t), (-t, t), (-t, 0), (-t, -t), (0, -t), (t, -t)]


# ── 运动：直线小步 + 逐步逆解检查 ─────────────────────────────────────────
class Mover:
    def __init__(self, rig, kin, keys, R0, q0):
        self.rig, self.kin, self.keys, self.R0 = rig, kin, keys, R0
        self.q = np.asarray(q0, float)          # dry-run 时就是规划出来的关节角
        self.psi = 0.0
        self.trace = [self.q]

    def pose(self):
        if self.rig is not None:
            self.q = self.rig.joints()
        return self.q, self.kin.fk(self.q)[:3, 3] * 1000

    def go(self, p, psi=None):
        """直线走到 p（mm），同时把绕竖直轴的转角插值到 psi。"""
        psi = self.psi if psi is None else float(psi)
        q, p0 = self.pose()
        p = np.asarray(p, float)
        n = max(1, int(np.ceil(max(np.linalg.norm(p - p0) / MOVE_STEP, abs(psi - self.psi) / YAW_STEP))))
        a0 = self.psi
        for k in range(1, n + 1):
            f = k / n
            pk, ak = p0 + (p - p0) * f, a0 + (psi - a0) * f
            qk, why = self.kin.solve(q, pk, Rz(ak) @ self.R0, HARD_MIN_Z)
            if qk is None:
                raise RuntimeError(f"走到 ({pk[0]:.0f},{pk[1]:.0f},{pk[2]:.0f}) 转角 {ak:.0f}° 不行：{why}")
            if self.rig is not None:
                self.rig.move(qk, self.keys)
            q = self.q = qk
            self.psi = ak
            self.trace.append(qk)

    def turn_about(self, c_xy, psi):
        """绕过 c_xy 的竖直轴把手转到 psi（位置沿圆弧）。法兰离转轴约 9cm，每 5° 一步单步关节会到 11°，所以每 2.5° 一步。"""
        _, p0 = self.pose()
        a0 = self.psi
        n = max(1, int(np.ceil(abs(psi - a0) / (YAW_STEP / 2))))
        c = np.asarray(c_xy, float)
        for k in range(1, n + 1):
            ak = a0 + (psi - a0) * k / n
            xy = c + Rz2(ak - a0) @ (p0[:2] - c)
            self.go([xy[0], xy[1], p0[2]], ak)


# ── mock：按「绕竖直轴转不变」模型合成腕部图，用来检验整套代数 ──────────────
class Sim:
    def __init__(self, spot, floor=286.0):
        self.floor = floor
        self.g = np.array([88.0, 4.0])                    # 真抓取点：套筒 − 法兰（ψ=0 基座系）
        self.S = None
        self.J_floor = np.array([[2.4, -2.3], [-2.45, -2.3]])

    def place(self, F_xy):                                # 「人」把套筒放在两指中间下方，放得不太准
        self.S = np.asarray(F_xy, float) + self.g + [3.0, -4.0]

    def k(self, z):
        return 110.0 / (110.0 + (z - self.floor))

    def image(self, p, psi):
        img = np.full((480, 640, 3), 235, np.uint8)
        if self.S is None:
            return img
        r = Rz2(-psi) @ (self.S - p[:2])
        c = np.array([304.0, 228.0]) + (p[2] - self.floor) * np.array([-0.1, -0.6])
        uv = c - self.J_floor * self.k(p[2]) @ (r - self.g)
        rad = 110 * self.k(p[2])
        cv2.circle(img, (int(uv[0]), int(uv[1])), int(rad), (200, 90, 20), -1)
        cv2.circle(img, (int(uv[0]), int(uv[1])), int(rad * 0.45), (40, 40, 40), -1)
        return img

    def close(self, p, psi):
        r = Rz2(-psi) @ (self.S - p[:2])
        held = p[2] <= self.floor + 3 and np.linalg.norm(r - self.g) < 9.0
        if held:                                          # 捏住会把套筒带到两指中间
            self.S = p[:2] + Rz2(psi) @ self.g + np.random.default_rng(0).normal(0, 0.5, 2)
        return held


# ── 拟合 ────────────────────────────────────────────────────────────────
def fit_layer(H, P):
    """P = c0 + J·H（H = 手相对参考点的水平偏移 mm，P = 套筒像素）。"""
    H, P = np.asarray(H, float), np.asarray(P, float)
    A = np.hstack([np.ones((len(H), 1)), H])
    X, *_ = np.linalg.lstsq(A, P, rcond=None)
    res = np.linalg.norm(A @ X - P, axis=1)
    return {"c0": X[0].tolist(), "J": X[1:].T.tolist(), "res_med": float(np.median(res)), "n": len(H)}


def solve_sleeve(layer, shots):
    """转腕 ψ 拍到的像素 p_ψ：p_ψ − c0 = J·(I − Rz(−ψ))·(S − F_ref) → 最小二乘解 S − F_ref。"""
    J, c0 = np.array(layer["J"]), np.array(layer["c0"])
    A = np.vstack([J @ (np.eye(2) - Rz2(-psi)) for psi, _ in shots])
    b = np.concatenate([np.asarray(p, float) - c0 for _, p in shots])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, float(np.abs(A @ x - b).max())


def dry_plan(kin, q_home, R0, spot, floor):
    m = Mover(None, kin, None, R0, q_home)
    _, ph = m.pose()
    s = np.asarray(spot, float)
    m.go([ph[0], ph[1], LIFT_Z])
    m.go([s[0], s[1], LIFT_Z])
    m.go([s[0], s[1], floor + 2])
    for dz in LAYERS:
        m.go([s[0], s[1], floor + dz])
        for dx, dy in grid_offsets():
            m.go([s[0] + dx, s[1] + dy, floor + dz])
        if dz == LAYERS[0]:
            m.go([s[0], s[1], floor + dz])
            for psi in YAWS:
                m.go([s[0], s[1], floor + dz], psi)
            m.go([s[0], s[1], floor + dz], 0)
    for ox, oy in trial_offsets():
        m.go([s[0] + ox, s[1] + oy, floor + LAYERS[1]])
        m.go([s[0] + ox, s[1] + oy, floor])
        m.go([s[0] + ox, s[1] + oy, floor + LAYERS[1]])
    m.go([s[0], s[1], floor + LAYERS[0]])
    for psi in YAWS:
        m.turn_about(s[:2] + [90, 0], psi)
    m.turn_about(s[:2] + [90, 0], 0)
    Q = np.vstack(m.trace)
    margin = np.degrees(np.minimum(Q - kin.lo, kin.hi - Q)).min()
    return len(Q), np.degrees(np.abs(np.diff(Q, axis=0))).max(), margin


# ── 主流程 ───────────────────────────────────────────────────────────────
def run(args):
    from atom_robot.ar5.config import AR5Config
    kin = CL.Kin()
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    T0 = kin.fk(q_home)
    R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
    tz = R0[:, 2]
    spot = np.asarray(args.spot, float)
    print(f"起始朝向：工具 z {tz.round(2)}（离竖直 {np.degrees(np.arccos(-tz[2])):.0f}°）；标定点 法兰 ({spot[0]:.0f},{spot[1]:.0f})，"
          f"指尖大约在它前方 9cm")
    worst = 0.0
    for fl in (270.0, 290.0, 310.0):
        try:
            n, jmax, mg = dry_plan(kin, q_home, R0, spot, fl)
            worst = max(worst, jmax)
            print(f"  假设下限 {fl:.0f}：全程 {n} 步可走，单步关节最多 {jmax:.1f}°，最小软限位余量 {mg:.1f}°")
        except RuntimeError as e:
            print(f"  假设下限 {fl:.0f}：✗ {e}")
    if args.dry_run:
        print("--dry-run：只算了路径，没连硬件。")
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    (out / "img").mkdir(parents=True, exist_ok=True)
    print(f"原始数据 → {out}")
    if not args.mock:
        print("=" * 70)
        print("  ⚠ 臂会：抬起 → 移到白板中间 → 你喊停量下限 → 你放套筒 → 自动拍网格、转腕、在套筒周围合手 9 次。")
        print("    开始前：白板上什么都不放，臂在起始位。第一次合手前会停下等你看。随时按 q 急停。")
        print("=" * 70)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    sim = Sim(spot) if args.mock else None
    rows, res = [], {"date": time.strftime("%Y-%m-%d %H:%M"), "R0": R0.tolist(), "spot": spot.tolist(),
                     "convention": ("同一层、ψ=0、套筒不动时：套筒像素 = c0 + J·(法兰xy − 参考点xy)。转 ψ 后像素对法兰的导数 = "
                                    "J·Rz(−ψ)。grasp_offset = 夹得住时 套筒 − 法兰（ψ=0 基座系 mm），转 ψ 后是 Rz(ψ)·grasp_offset。"
                                    "ring = 下限高度、对准抓取点时套筒在腕部画面里的中心和半径。")}

    def grab():
        if sim is not None:
            _, p = mover.pose()
            return {"cam_front": sim.image(p, mover.psi), "cam_high": np.zeros((480, 640, 3), np.uint8),
                    "cam_side": np.zeros((480, 640, 3), np.uint8)}
        return CL.grab(rig)

    def shoot(tag, show_text=None):
        fr = grab()
        _, p = mover.pose()
        n = len(rows)
        for k, sub in (("cam_front", "wrist"), ("cam_high", "top"), ("cam_side", "side")):
            if fr.get(k) is not None:
                cv2.imwrite(str(out / "img" / f"{n:03d}_{tag}_{sub}.jpg"), fr[k])
        b = blob(fr.get("cam_front"))
        rows.append({"n": n, "tag": tag, "flange": p.round(2).tolist(), "psi": mover.psi,
                     "blob": None if b is None else [round(v, 1) for v in b]})
        if not args.mock:
            w = fr["cam_front"].copy()
            if b is not None:
                cv2.circle(w, (int(b[0]), int(b[1])), int(b[2]), (0, 255, 0), 2)
            CL.show([fr.get("cam_high", w), w] + ([fr["cam_side"]] if "cam_side" in fr else []),
                    show_text or f"{tag}  flange ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})  yaw {mover.psi:.0f}")
        return b, p

    def ask(prompt, mock_answer=""):
        if args.mock:
            return mock_answer
        return keys.ask(prompt)

    def save():
        res["shots"] = rows
        (out / "meta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    t0 = time.time()
    try:
        rig.connect()
        if np.degrees(np.abs(rig.joints() - q_home)).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在起始位。先回位（灵巧手遥操/VR设备/home_arm.py）再跑")
        with rc.Keys(mock=args.mock) as keys:
            mover = Mover(rig, kin, keys, R0, rig.joints())
            rig.set_hand("open")

            # 1 下限 ------------------------------------------------------
            print("\n1 抬起、移到标定点上方")
            mover.go([p_home[0], p_home[1], LIFT_Z])
            mover.go([spot[0], spot[1], LIFT_Z])
            shoot("spot")
            if ask("  看一眼：所有指尖正下方都是白板、白板上什么都没有？是就回车，q 中止: ") == "q":
                raise rc.StopRequested("量下限前中止")
            step = 10.0
            while args.floor is None:
                _, p = mover.pose()
                ans = ask(f"  法兰 z={p[2]:.1f}mm。回车=再降 {step:g}mm，输数字=改步长并降，u=抬 5mm，"
                          f"ok=所有指尖里最低的那根（可能是中指/无名指/小指）离板 3~5mm 了: ",
                          "ok" if sim is not None and p[2] <= sim.floor + 0.5 else "")
                if ans == "ok":
                    floor = float(p[2])
                    break
                if ans == "q":
                    raise rc.StopRequested("量下限时中止")
                if ans == "u":
                    mover.go(p + [0, 0, 5])
                    continue
                if ans:
                    try:
                        step = float(np.clip(float(ans), 0.5, 10.0))
                    except ValueError:
                        print("  没看懂，再输一次")
                        continue
                if p[2] - step < HARD_MIN_Z:
                    print(f"  再降就低于硬下限 {HARD_MIN_Z:.0f}mm，不降。指尖还没到板子的话，告诉我调这个值")
                    continue
                mover.go(p - [0, 0, step])
                shoot("floor_probe")
            if args.floor is not None:
                floor = float(args.floor)
                print(f"  用给定的法兰下限 {floor:.1f}mm（不再量）")
            res["floor"] = floor
            print(f"  → 法兰下限 {floor:.1f}mm")
            save()

            # 2 放套筒，并在下限试合一次确认放对了 ------------------------------
            # 10-13 第一次真跑：让人把套筒放在「张开时两指尖中间」下方，9 次合手全空（手指读数 = 指令值，根本没碰到）。
            # 合拢时指尖会往手掌方向卷，真正的捏合点比张开时的指尖中点更靠里；所以先合一次给人看捏合点，放好后实际试合确认。
            z90, z60 = floor + LAYERS[0], floor + LAYERS[1]
            F_ref = spot[:2].copy()
            mover.go([F_ref[0], F_ref[1], z90])
            print("\n2 放套筒：手先在空中合一次，看拇指和食指指尖碰到的位置")
            rig.set_hand("close")
            ask("  记住两指尖碰到的那一点，回车张手: ")
            rig.set_hand("open")
            for k in range(6):
                if sim is not None:
                    sim.place(F_ref)
                ask("  把蓝套筒立在白板上，放在刚才两指尖碰到的那一点正下方（比张开时的指尖更靠手掌），放好回车: ")
                b, _ = shoot("placed")
                if b is None:
                    print("  腕部画面里没找到蓝色套筒，再放一次")
                    continue
                if ask("  手要降到下限试合一次。看着手指别压到套筒上，回车继续，q 中止: ") == "q":
                    raise rc.StopRequested("放套筒试合前中止")
                mover.go([F_ref[0], F_ref[1], floor])
                _, pf = shoot("place_floor")
                ang = rig.set_hand("close")
                held, ex_i, ex_t = CL.finger_held(ang)
                if sim is not None:
                    held = sim.close(pf, mover.psi)
                rig.set_hand("open")
                mover.go([F_ref[0], F_ref[1], z90])
                print(f"  试合：食指 {ang[rc.INDEX]}（+{ex_i}）拇指 {ang[rc.THUMB_BEND]}（+{ex_t}）→ "
                      f"{'夹住了，套筒已被捏到捏合点' if held else '空夹'}")
                res.setdefault("place_tries", []).append({"fingers": [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])],
                                                          "held": bool(held)})
                if held:
                    break
                print("  手指没碰到套筒：对照侧面相机和实物，把套筒往捏合点挪一挪再试")
            else:
                raise RuntimeError("放了 6 次试合都没夹住：这个朝向可能抓不到套筒（捏合点太高或够不着）")
            save()

            # 3 网格 + 4 转腕 ---------------------------------------------
            layers = {}
            for dz in LAYERS:
                z = floor + dz
                print(f"\n3 网格 +{dz:.0f}mm（只拍照）")
                mover.go([F_ref[0], F_ref[1], z])
                if dz == LAYERS[1] and ask("  看一眼：指尖要高过套筒顶。是就回车，不是输入 q: ") == "q":
                    raise rc.StopRequested("+60 层指尖不够高")
                H, P = [], []
                for dx, dy in grid_offsets():
                    mover.go([F_ref[0] + dx, F_ref[1] + dy, z])
                    b, p = shoot(f"G{dz:.0f}")
                    if b is not None:
                        H.append(p[:2] - F_ref)
                        P.append(b[:2])
                mover.go([F_ref[0], F_ref[1], z])
                if len(H) < 15:
                    raise RuntimeError(f"+{dz:.0f} 层只有 {len(H)}/25 张找到套筒，拟合不了")
                layers[f"{dz:.0f}"] = fit_layer(H, P)
                print(f"  拟合 {len(H)} 点，残差中位 {layers[f'{dz:.0f}']['res_med']:.1f}px")
                if dz == LAYERS[0]:
                    print(f"\n4 转腕 {YAWS}°（原地绕竖直轴，只拍照）")
                    shots = []
                    for psi in YAWS:
                        mover.go([F_ref[0], F_ref[1], z], psi)
                        b, _ = shoot(f"yaw{psi:.0f}")
                        if b is not None:
                            shots.append((psi, b[:2]))
                    mover.go([F_ref[0], F_ref[1], z], 0.0)
                    if not shots:
                        raise RuntimeError("转腕后腕部画面里看不到套筒")
                    x, resid = solve_sleeve(layers[f"{dz:.0f}"], shots)
                    S0 = F_ref + x
                    res["sleeve_rel_at_place"] = x.tolist()
                    print(f"  套筒在法兰的 ({x[0]:+.0f},{x[1]:+.0f})mm（{len(shots)} 个角度，残差 {resid:.1f}px）")
                    if sim is not None:
                        print(f"  [mock 真值 ({sim.S[0] - F_ref[0]:+.0f},{sim.S[1] - F_ref[1]:+.0f})]")
            res["layers"] = layers
            save()

            # 5 合手 ------------------------------------------------------
            L60 = layers[f"{LAYERS[1]:.0f}"]
            J60, c060 = np.array(L60["J"]), np.array(L60["c0"])
            r_guess = S0 - F_ref
            area0, trials = None, []
            print(f"\n5 合手试 {len(trial_offsets())} 次（±{TRIAL:.0f}mm）")
            for i, (ox, oy) in enumerate(trial_offsets()):
                mover.go([F_ref[0], F_ref[1], z60])
                for _ in range(3):
                    b, _ = shoot(f"T{i}_ref")
                    ok = b is not None and (area0 is None or abs(b[3] / area0 - 1) < 0.4)
                    if ok:
                        break
                    ask("  套筒不见了或者倒了/被推远了：把它立好，放回指尖中间下方，回车: ")
                    if sim is not None:
                        sim.place(F_ref)
                else:
                    raise RuntimeError("参考点一直找不到套筒")
                area0 = area0 or b[3]
                S_i = S0 - np.linalg.solve(J60, np.asarray(b[:2]) - c060)
                r_i = r_guess + [ox, oy]
                F_i = S_i - r_i
                mover.go([F_i[0], F_i[1], z60])
                if i == 0 and ask("  第一次：要降到下限合手了。看着指尖，确认会落在套筒两侧就回车，q 中止: ") == "q":
                    raise rc.StopRequested("第一次合手前中止")
                mover.go([F_i[0], F_i[1], floor])
                bf, pf = shoot(f"T{i}_floor")
                ang = rig.set_hand("close")
                held, ex_i, ex_t = CL.finger_held(ang)
                if sim is not None:
                    held = sim.close(pf, mover.psi)
                rig.set_hand("open")
                mover.go([F_i[0], F_i[1], z60])
                trials.append({"offset": [ox, oy], "r": r_i.tolist(), "held": bool(held),
                               "fingers": [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])],
                               "floor_blob": None if bf is None else [round(v, 1) for v in bf]})
                print(f"  {i + 1}/9 偏 ({ox:+.0f},{oy:+.0f})：食指 {ang[rc.INDEX]} 拇指 {ang[rc.THUMB_BEND]} → "
                      f"{'夹住' if held else '空'}")
                res["trials"] = trials
                save()

            held_r = np.array([t["r"] for t in trials if t["held"]])
            print("\n  夹得住的分布（行 = 偏移 y +10/0/−10，列 = 偏移 x −10/0/+10）：")
            tab = {tuple(t["offset"]): t["held"] for t in trials}
            for oy in (TRIAL, 0, -TRIAL):
                print("    " + "  ".join("✓" if tab[(ox, oy)] else "✗" for ox in (-TRIAL, 0, TRIAL)))
            if len(held_r) == 0:
                res["grasp_offset_mm"] = None
                res["ring"] = None
                print("  ✗ 9 次一次都没夹住：这个朝向可能根本抓不了套筒，环和转轴都没法定。先别用它跑 auto_loop。")
            else:
                g = held_r.mean(0)
                res["grasp_offset_mm"] = g.tolist()
                fb = [(t["r"], t["floor_blob"]) for t in trials if t["floor_blob"] is not None]
                if len(fb) >= 4:              # 下限高度上：像素 = A + B·r，取 r = g 处
                    R_ = np.array([np.r_[1.0, r] for r, _ in fb])
                    X, *_ = np.linalg.lstsq(R_, np.array([f[:2] for _, f in fb]), rcond=None)
                    centre = X[0] + X[1:].T @ g
                else:
                    centre = np.mean([t["floor_blob"][:2] for t in trials if t["held"] and t["floor_blob"]], 0)
                rad = float(np.median([f[2] for _, f in fb])) if fb else 110.0
                res["ring"] = {"center": [round(float(v), 1) for v in centre], "radius": round(rad, 1)}
                print(f"  抓取点 = 法兰 + ({g[0]:+.1f},{g[1]:+.1f})mm；环 中心 ({centre[0]:.0f},{centre[1]:.0f}) 半径 {rad:.0f}px")
                if sim is not None:
                    print(f"  [mock 真值 抓取点 ({sim.g[0]:+.0f},{sim.g[1]:+.0f})、环中心 (304,228)、半径 110]")
            save()

            # 6 验证转腕 ---------------------------------------------------
            if res.get("grasp_offset_mm") is not None:
                print("\n6 验证：对准套筒，绕抓取点转腕")
                mover.go([F_ref[0], F_ref[1], z60])
                b, _ = shoot("V_ref")
                S_now = S0 - np.linalg.solve(J60, np.asarray(b[:2]) - c060)
                F_al = S_now - g
                mover.go([F_al[0], F_al[1], z90])
                b0, _ = shoot("V0")
                drift = []
                for psi in YAWS:
                    mover.turn_about(S_now, psi)
                    bv, _ = shoot(f"V{psi:.0f}")
                    if b0 is not None and bv is not None:
                        drift.append((psi, float(np.hypot(bv[0] - b0[0], bv[1] - b0[1]))))
                mover.turn_about(S_now, 0.0)
                res["yaw_check_px"] = drift
                print("  套筒在画面里移动：" + "，".join(f"{p:.0f}° → {d:.0f}px" for p, d in drift)
                      + "（+90 层 1mm ≈ 2px；绕法兰转的话会移一百多 px）")
                save()

            # 7 回起始位 ----------------------------------------------------
            print("\n7 回起始位")
            _, p = mover.pose()
            mover.go([p[0], p[1], LIFT_Z], 0.0)
            mover.go([p_home[0], p_home[1], LIFT_Z])
            mover.go(p_home)
            rig.move(q_home, keys)
            res["complete"] = res.get("ring") is not None
            # mock 的假数据、没定出环的标定都不能盖掉真标定
            dst = CALIB_FILE if res["complete"] and not args.mock else out / CALIB_FILE.name
            dst.write_text(json.dumps({k: v for k, v in res.items() if k != "shots"}, ensure_ascii=False,
                                      indent=1), encoding="utf-8")
            print(f"\n标定完成（{time.time() - t0:.0f}s）→ {dst}")
    except rc.StopRequested as e:
        print(f"\n停下：{e}。臂停在原地，手 {rig.hand_state}。")
    except Exception as e:  # noqa: BLE001
        print(f"\n出错停下：{type(e).__name__}: {e}。臂停在原地，手 {rig.hand_state}。")
    finally:
        save()
        print(f"原始数据 → {out}")
        cv2.destroyAllWindows()
        rig.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", type=float, nargs=2, default=(340.0, -160.0),
                    help="标定点的法兰 xy（mm）；起始朝向下指尖在它前方约 9cm，要落在白板中间")
    ap.add_argument("--floor", type=float, default=None, help="已量过的法兰下限（mm），给了就跳过人工量下限")
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_calib")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
