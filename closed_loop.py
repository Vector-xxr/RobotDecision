#!/usr/bin/env python3
"""RobotDecision 真机闭环（人在回路）：每一步由 OpenAI Decisions 看图选动作，人按回车才执行。

策略（10-13 离线评测定下来的拆题方式；用哪一题由上一题的答案决定，动作都由 Decisions 选）
  手张开时
    1. 腕部专题（只给腕部图：绿圈 + 4 根同字母箭头）：哪根箭头从圈指向套筒 / centered / not_visible
       —— 离线 77%，镜像 72%
    2. not_visible → 俯视题（俯视图 + 箭头，腕部图 + 圈）：选箭头，或 target_not_visible —— 远处 88~94%
    3. centered → 双圈题（实圈 + 虚线内圈）：≥0.2 合手，否则下降 —— 离线提前合手 0/72
  手合上后：按规则抬 1cm/步共 8cm，每步问「还夹着吗」；抬够了 = 完成
  同一道选择题正序、倒序各问一次，两次不一致 → 这一步不动、重新拍重新问

动作 = 基座系平移（保持抓取时的手部朝向），用自己的 PoE 逆解 + MoveAbsJ，实测关节到位才算完成。
安全边界（物理的，不是画框）：逆解可达、软限位余量、单步关节 ≤10°、法兰不低于 --floor（默认示范抓取高度 −10mm）。

    python closed_loop.py --dry-run     # 只算起步路径和姿态，不连硬件、不调 API
    python closed_loop.py --mock        # 假臂 + 假手 + 假相机 + 假 Decisions，走一遍流程
    python closed_loop.py               # 真跑：每一步回车确认
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
import near_tests as NT  # noqa: E402

START_Z = 340.0          # 起步：沿示范走到第一个法兰 z ≥ 这个值的点（已在白板上方）
STEP_FAR, STEP_NEAR = 20.0, 10.0
LOWER_BIG, LOWER_SMALL, RAISE = 10.0, 5.0, 10.0
LIFT_STEPS = 8
MAX_JUMP_DEG = 10.0
CLOSE_P = 0.2            # 双圈题：≥0.2 合手（离线该合 8/10、提前合 0/72）
MAX_STEPS = 80
DIR_BASE = {"north": (1, 0, 0), "south": (-1, 0, 0), "west": (0, 1, 0), "east": (0, -1, 0)}


# ── 运动学 ────────────────────────────────────────────────────────────────
class Kin:
    def __init__(self):
        from ar5_poe_ik import AR5PoeIK
        self.ik = AR5PoeIK(max_step=np.pi, refine_iters=10)
        self.fk = self.ik.cal.fk
        a = rc_cfg_arm()
        self.lo = np.asarray(a.joint_min) + a.joint_limit_margin
        self.hi = np.asarray(a.joint_max) - a.joint_limit_margin

    def solve(self, q_now, p_mm, R, floor):
        """返回 (q, 原因)；q=None 表示拒绝。"""
        if p_mm[2] < floor:
            return None, f"法兰会低到 z={p_mm[2]:.0f}mm，低于下限 {floor:.0f}mm"
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(p_mm) / 1000.0
        q, info = self.ik.ik(T, q_now)
        if q is None:
            return None, f"逆解不可达：{info.get('reason')}"
        Tc = self.fk(q)
        if np.linalg.norm(Tc[:3, 3] * 1000 - p_mm) > 1.0:
            return None, "逆解回代差 >1mm"
        if np.any(q < self.lo) or np.any(q > self.hi):
            return None, "超软限位"
        jump = np.degrees(np.abs(q - q_now)).max()
        if jump > MAX_JUMP_DEG:
            return None, f"一步关节要转 {jump:.1f}°（>{MAX_JUMP_DEG}°）"
        return q, ""


def rc_cfg_arm():
    from atom_robot.ar5.config import AR5Config
    return AR5Config(mock=True).arm


def axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax, ang


def rot(ax, ang):
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def start_plan(args, kin):
    """沿示范走到第一个 z ≥ START_Z 的点（已在白板上方），再原地每 5° 一步转到抓取朝向。
    10-13 dry-run：直接一步转过去关节要转 59.9°，所以拆成小步，每步仍过 MAX_JUMP_DEG 检查。"""
    plan = rc.make_plan(args)
    P, q, g = plan["P"], plan["q"], plan["i_grasp"]
    R_g = plan["T"][g][:3, :3]
    path = [i for i in plan["transit"] if i <= g]
    k = next(j for j, i in enumerate(path) if P[i, 2] >= START_Z)
    lead = path[:k + 1]
    # 10-13：原地转朝向在第 5/6 步逆解换支（关节跳 54.5°）——那个位置上抓取朝向让 j7 贴限位。
    # 改成位置和朝向一起插值到 --start（白板右半边，j7 余量大；全程最小余量 8.2°、单步 ≤2.7°）。
    R_wp, p_wp = plan["T"][lead[-1]][:3, :3], P[lead[-1]]
    p_st = np.asarray(args.start, float)
    ax, ang = axis_angle(R_wp.T @ R_g)
    n = 30
    q_rot, qs = [], q[lead[-1]]
    for s in range(1, n + 1):
        f = s / n
        qs, why = kin.solve(qs, p_wp + (p_st - p_wp) * f, R_wp @ rot(ax, ang * f), -1e9)
        if qs is None:
            return plan, lead, None, f"过渡第 {s}/{n} 步：{why}", R_g
        q_rot.append(qs)
    return plan, lead, q_rot, f"转 {np.degrees(ang):.1f}°、平移到 {tuple(int(v) for v in p_st)}，分 {n} 步", R_g


# ── 拍照与提问 ────────────────────────────────────────────────────────────
def grab(rig):
    time.sleep(rc.SETTLE_S)
    fr = rig.cams.read()
    return {k: cv2.cvtColor(v, cv2.COLOR_RGB2BGR) for k, v in fr.items() if v is not None}


def url(img):
    return NT.img_url(img)


def ask_wrist(dec, wrist, letters):
    img = wrist.copy()
    (cu, cv_), rad = E.WRIST_CIRCLE
    cv2.circle(img, (cu, cv_), rad, (0, 0, 0), 6)
    cv2.circle(img, (cu, cv_), rad, (0, 255, 0), 3)
    E.draw_wrist_arrows(img, (cu, cv_), letters, False)
    ch = [{"value": f"arrow_{k}", "description": f"Arrow {k}."} for k in "ABCD"]
    ch += [{"value": "centered", "description": "The sleeve is already centered on the green circle."},
           {"value": "not_visible", "description": "No blue sleeve is visible in the image."}]
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": NT.A1_TEXT},
                                                  {"type": "input_image", "image_url": url(img), "detail": "auto"}]}],
           "questions": [{"type": "choice", "name": "a", "instructions": NT.A1_Q, "choices": ch},
                         {"type": "choice", "name": "b", "instructions": NT.A1_Q, "choices": ch[::-1]}]}
    res = dec(req, "wrist")
    return res, img


def ask_top(dec, top, wrist, flange_xy, letters):
    t = top.copy()
    E.draw_arrows(t, E.TOP_MAP(flange_xy), letters)
    w = wrist.copy()
    (cu, cv_), rad = E.WRIST_CIRCLE
    cv2.circle(w, (cu, cv_), rad, (0, 255, 0), 3)
    text = E.CONTEXT_V3 + "\nCurrent hand state: open."
    ch = [{"value": f"arrow_{k}", "description": f"Move the hand a short step along arrow {k} drawn in image 1."}
          for k in "ABCD"]
    ch += [{"value": "target_not_visible", "description": "The blue sleeve cannot be seen in either image."}]
    q = "Which single action should the robot take next to move the hand toward the blue sleeve?"
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": text},
                                                  {"type": "input_text", "text": "Image 1 (top camera):"},
                                                  {"type": "input_image", "image_url": url(t), "detail": "auto"},
                                                  {"type": "input_text", "text": "Image 2 (wrist camera):"},
                                                  {"type": "input_image", "image_url": url(w), "detail": "auto"}]}],
           "questions": [{"type": "choice", "name": "a", "instructions": q, "choices": ch},
                         {"type": "choice", "name": "b", "instructions": q, "choices": ch[::-1]}]}
    res = dec(req, "top")
    return res, t


def ask_rings(dec, wrist):
    img = wrist.copy()
    (cu, cv_), rad = E.WRIST_CIRCLE
    cv2.circle(img, (cu, cv_), rad, (0, 0, 0), 6)
    cv2.circle(img, (cu, cv_), rad, (0, 255, 0), 3)
    for k in range(0, 360, 20):
        cv2.ellipse(img, (cu, cv_), (NT.INNER_R, NT.INNER_R), 0, k, k + 10, (0, 255, 0), 2)
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": NT.A2_RINGS_TEXT},
                                                  {"type": "input_image", "image_url": url(img), "detail": "auto"}]}],
           "questions": [{"type": "predicate", "name": "fills", "instructions": NT.A2_RINGS_Q}]}
    res = dec(req, "rings")
    return res, img


SIDE_Q_TEXT = ("This image is from a camera at table level, looking across a white board at a blue sleeve "
               "standing on it. The dark fingers of a robot hand come down from above.")
SIDE_Q = ("Are the robot's fingertips already down at the height of the blue sleeve (below its top edge) and on "
          "both sides of it, so that closing the fingers now would grasp the sleeve?")


def ask_side(dec, side):
    """10-13 闭环空夹：腕部双圈题 0.43 就合了，侧面图里指尖明明还在套筒顶上方。合手前加这一题。"""
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": SIDE_Q_TEXT},
                                                  {"type": "input_image", "image_url": url(side), "detail": "auto"}]}],
           "questions": [{"type": "predicate", "name": "ready", "instructions": SIDE_Q}]}
    return dec(req, "side")


def finger_held(ang):
    """手指被挡住 = 夹住了（和 replay_capture 同一判据）。10-13 闭环空夹时实测 500/400 = 指令值，
    而「还夹着吗」那道题答了 0.90/0.95 —— 看图判抓没抓住不可靠，用手指自己的读数。"""
    idx_cmd = max(rc.HAND_CLOSE[0], rc_ih().index_floor(rc.HAND_CLOSE[1]))
    ex_i, ex_t = ang[rc.INDEX] - idx_cmd, ang[rc.THUMB_BEND] - rc.HAND_CLOSE[1]
    return (ex_i > 40 and ex_t > 40) or max(ex_i, ex_t) > 150, ex_i, ex_t


def rc_ih():
    import inspire_hand6 as ih
    return ih


def ask_holding(dec, wrist):
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [
               {"type": "input_text", "text": "This image is from a camera fixed on a robot hand, looking down past "
                                              "the robot's thumb and index finger."},
               {"type": "input_image", "image_url": url(wrist), "detail": "auto"}]}],
           "questions": [{"type": "predicate", "name": "holding",
                          "instructions": "Is the blue sleeve held between the thumb and the index finger?"}]}
    return dec(req, "holding")


def make_decider(args, log_dir):
    if args.mock:
        rnd = random.Random(0)

        def fake(req, kind):
            if kind == "wrist":
                v = rnd.choice(["arrow_A", "arrow_B", "centered", "centered", "not_visible"])
                return {"answers": {"a": {"choice": v}, "b": {"choice": v}}, "ms": 1.0, "input_tokens": 0}
            if kind == "top":
                v = rnd.choice(["arrow_A", "arrow_C"])
                return {"answers": {"a": {"choice": v}, "b": {"choice": v}}, "ms": 1.0, "input_tokens": 0}
            if kind == "side":
                return {"answers": {"ready": {"probability": rnd.choice([0.1, 0.8])}}, "ms": 1.0, "input_tokens": 0}
            if kind == "rings":
                return {"answers": {"fills": {"probability": rnd.choice([0.0, 0.1, 0.6])}}, "ms": 1.0, "input_tokens": 0}
            return {"answers": {"holding": {"probability": 0.9}}, "ms": 1.0, "input_tokens": 0}
        return fake
    from openai import OpenAI
    client = OpenAI(api_key=E.KEY_FILE.read_text().strip(), max_retries=0)
    throttle = E.Throttle(120)
    counter = {"n": 0}

    def real(req, kind):
        res = E.call(client, req, throttle)
        counter["n"] += 1
        (log_dir / f"api_{counter['n']:03d}_{kind}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res
    return real


# ── 主循环 ────────────────────────────────────────────────────────────────
def show(panels, text, display=True):
    h = 300
    tiles = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    img = np.hstack(tiles)
    bar = np.zeros((70, img.shape[1], 3), np.uint8)
    for i, line in enumerate(text.split("\n")[:3]):
        cv2.putText(bar, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    out = np.vstack([img, bar])
    if display:
        cv2.imshow("RobotDecision closed loop", out)
        cv2.waitKey(100)
    return out


def confirm(args, keys, prompt):
    if args.mock:
        print(prompt + "（mock：自动执行）")
        return ""
    return keys.ask(prompt)


def run(args):
    kin = Kin()
    plan, lead, q_rot, why, R_g = start_plan(args, kin)
    q_start = q_rot[-1] if q_rot else None
    P, q, g = plan["P"], plan["q"], plan["i_grasp"]
    floor = P[g, 2] - 10.0 if args.floor is None else args.floor
    print(f"起步：沿示范走 {len(lead)} 点到 t={plan['t'][lead[-1]]:.1f}s 法兰 "
          f"({P[lead[-1],0]:.0f},{P[lead[-1],1]:.0f},{P[lead[-1],2]:.0f})，再过渡到起步位姿（{why}）"
          + (f"，每步关节最多 {max(np.degrees(np.abs(np.diff(np.vstack([q[lead[-1]], *q_rot]), axis=0))).max(1)):.1f}°" if q_rot else ""))
    if q_start is None:
        print(f"  ✗ 起步姿态解不出：{why}")
        return 1
    print(f"法兰下限 {floor:.0f}mm（示范抓取高度 {P[g,2]:.0f}mm −10）；步长 远 {STEP_FAR:.0f} / 近 {STEP_NEAR:.0f} / "
          f"降 {LOWER_BIG:.0f}~{LOWER_SMALL:.0f} / 抬 {RAISE:.0f} mm；最多 {MAX_STEPS} 步")
    if args.dry_run:
        print("--dry-run：没连硬件、没调 API。")
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"记录 → {out}")
    if not args.mock:
        print("=" * 66)
        print("  ⚠ 臂会先沿示范抬到白板上方，然后每一步等你回车才动。")
        print("    开始前：套筒放在白板上（任意位置），臂在起始位。")
        print("    每步：回车=执行  s=不执行、重新拍重新问  q=停（臂停在原地）")
        print("=" * 66)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    args_rc = argparse.Namespace(mock=args.mock, speed=args.speed)
    rig = rc.Rig(args_rc, out)
    dec = make_decider(args, out)
    log = []
    rng = np.random.default_rng(int(time.time()))
    try:
        rig.connect()
        if np.degrees(np.abs(rig.joints() - q[0])).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在示范起点。先 python home_arm.py 回位再跑")
        with rc.Keys(mock=args.mock) as keys:
            rig.set_hand("open")
            print("\n起步：沿示范抬到白板上方 ...")
            for i in lead[1:]:
                rig.move(q[i], keys)
            for qq in q_rot:
                rig.move(qq, keys)
            closed, lifted, step = False, 0, 0
            while step < MAX_STEPS:
                step += 1
                qn = rig.joints()
                Tn = kin.fk(qn)
                pn = Tn[:3, 3] * 1000
                fr = grab(rig)
                top, wrist, side = fr.get("cam_high"), fr.get("cam_front"), fr.get("cam_side")
                rec = {"step": step, "flange": pn.round(1).tolist(), "closed": closed}
                cv2.imwrite(str(out / f"{step:03d}_top.jpg"), top)
                cv2.imwrite(str(out / f"{step:03d}_wrist.jpg"), wrist)
                if side is not None:
                    cv2.imwrite(str(out / f"{step:03d}_side.jpg"), side)

                action, delta, panels, why_txt = None, None, [], ""
                top_ann, wr_ann = top, wrist        # 窗口固定三路，问了哪题就换成那题的标注图
                if closed:
                    ang = rig.hand.get_angles()
                    held, ex_i, ex_t = finger_held(ang)
                    if args.mock:
                        held = not args.mock_empty
                    rec["fingers"] = [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])]
                    if not held:
                        action, why_txt = "stop", f"手指没被挡住（食指 {ang[rc.INDEX]} / 拇指 {ang[rc.THUMB_BEND]}）= 掉了"
                    elif lifted >= LIFT_STEPS:
                        action, why_txt = "done", f"已抬 {lifted * RAISE:.0f}mm，手指仍被挡住（+{ex_i}/+{ex_t}）"
                    else:
                        action, delta, why_txt = "raise", (0, 0, RAISE), f"手指被挡住（+{ex_i}/+{ex_t}），按规则继续抬"
                    panels = [top, wrist]
                else:
                    letters = dict(zip("ABCD", rng.permutation(["north", "east", "south", "west"])))
                    res, wimg = ask_wrist(dec, wrist, letters)
                    wr_ann = wimg
                    a, b = res["answers"]["a"]["choice"], res["answers"]["b"]["choice"]
                    rec["wrist"] = [a, b]
                    panels = [wimg]
                    if a != b:
                        action, why_txt = "reask", f"腕部题两次不一致 {a}/{b}"
                    elif a.startswith("arrow_"):
                        d = letters[a[-1]]
                        action, delta, why_txt = f"move_{d}", np.array(DIR_BASE[d]) * STEP_NEAR, f"腕部题：{a} = {d}"
                    elif a == "not_visible":
                        res2, timg = ask_top(dec, top, wrist, pn[:2], letters)
                        top_ann = timg
                        a2, b2 = res2["answers"]["a"]["choice"], res2["answers"]["b"]["choice"]
                        rec["top"] = [a2, b2]
                        panels = [timg, wimg]
                        if a2 != b2:
                            action, why_txt = "reask", f"俯视题两次不一致 {a2}/{b2}"
                        elif a2 == "target_not_visible":
                            action, why_txt = "stop", "两路都说看不到套筒"
                        else:
                            d = letters[a2[-1]]
                            action, delta, why_txt = f"move_{d}", np.array(DIR_BASE[d]) * STEP_FAR, f"俯视题：{a2} = {d}"
                    else:      # centered
                        res3, rimg = ask_rings(dec, wrist)
                        wr_ann = rimg
                        p = res3["answers"]["fills"]["probability"]
                        rec["rings"] = p
                        panels = [wimg, rimg]
                        if p >= CLOSE_P and side is not None:
                            res4 = ask_side(dec, side)
                            ps = res4["answers"]["ready"]["probability"]
                            rec["side_ready"] = ps
                            if ps >= 0.5:
                                action, why_txt = "close", f"已对正，双圈题 {p:.2f} ≥ {CLOSE_P}，侧面：指尖已到套筒两侧 {ps:.2f}"
                            else:
                                action, delta, why_txt = "lower", (0, 0, -LOWER_SMALL), (
                                    f"双圈题 {p:.2f} 够了，但侧面：指尖还没降到套筒两侧 {ps:.2f} → 下降 {LOWER_SMALL:.0f}mm")
                        elif p >= CLOSE_P:
                            action, why_txt = "close", f"已对正，双圈题 {p:.2f} ≥ {CLOSE_P}（没有侧面图）"
                        else:
                            dz = LOWER_BIG if p < 0.05 else LOWER_SMALL
                            action, delta, why_txt = "lower", (0, 0, -dz), f"已对正，双圈题 {p:.2f} < {CLOSE_P} → 下降 {dz:.0f}mm"
                panels = [top_ann, wr_ann] + ([side] if side is not None else [])

                q_target = None
                if delta is not None:
                    p_new = pn + np.asarray(delta, float)
                    q_target, reason = kin.solve(qn, p_new, R_g, floor)
                    if (q_target is None and action == "lower" and "下限" in reason
                            and (rec.get("rings") or 0) >= CLOSE_P):
                        # 侧面题还没确认过「能抓」（没真机正例验证过），但已经降到下限、双圈题也够了：
                        # 提出合手，由人看侧面图决定回车与否，不卡死在这里
                        action, delta = "close", None
                        why_txt += f"；已到法兰下限 {floor:.0f}mm、双圈题够了 → 提议合手（侧面没确认，请看侧面图再决定）"
                    elif q_target is None:
                        why_txt += f"；✗ 不能执行：{reason}"
                        action = action + "（被拒）"
                txt = (f"step {step}  flange ({pn[0]:.0f},{pn[1]:.0f},{pn[2]:.0f})  -> {action}\n{why_txt}")
                print("\n" + txt.replace("\n", "\n  "))
                shot = show(panels, f"step {step}  -> {action}\n{why_txt[:90]}", display=not args.mock)
                cv2.imwrite(str(out / f"{step:03d}_panel.jpg"), shot)
                rec.update(action=action, why=why_txt)

                if action in ("stop", "done"):
                    rec["user"] = "-"
                    log.append(rec)
                    print(f"  → {'完成' if action == 'done' else '停止'}：{why_txt}")
                    break
                if action == "reask" or q_target is None and action != "close":
                    rec["user"] = "auto-skip"
                    log.append(rec)
                    continue
                ans = confirm(args, keys, "  回车=执行  s=跳过重问  q=停 > ")
                rec["user"] = ans or "enter"
                log.append(rec)
                if ans == "q":
                    raise rc.StopRequested("操作者按 q")
                if ans == "s":
                    continue
                if action == "close":
                    ang = rig.set_hand("close")
                    held, ex_i, ex_t = finger_held(ang)
                    if args.mock:
                        held = not args.mock_empty
                    print(f"  合手：食指 {ang[rc.INDEX]} / 拇指弯 {ang[rc.THUMB_BEND]}（比指令多 +{ex_i}/+{ex_t}）→ "
                          + ("被挡住 = 夹住了" if held else "没被挡住 = 空夹"))
                    rec["grasp"] = {"index": int(ang[rc.INDEX]), "thumb": int(ang[rc.THUMB_BEND]), "held": bool(held)}
                    if held:
                        closed = True
                    else:
                        rig.set_hand("open")
                        qn = rig.joints()
                        pn = kin.fk(qn)[:3, 3] * 1000
                        qt, reason = kin.solve(qn, pn + [0, 0, 20], R_g, floor)
                        if qt is not None:
                            rig.move(qt, keys)
                        print("  空夹：已张手" + ("、抬高 2cm，继续" if qt is not None else f"；抬不上去（{reason}），继续"))
                else:
                    rig.move(q_target, keys)
                    if closed and action == "raise":
                        lifted += 1

            if closed:
                ans = confirm(args, keys, "\n  放下并退回起点？回车=是  q=停在这里 > ")
                if ans != "q":
                    for _ in range(lifted):
                        qn = rig.joints()
                        pn = kin.fk(qn)[:3, 3] * 1000
                        qt, reason = kin.solve(qn, pn - [0, 0, RAISE], R_g, floor)
                        if qt is None:
                            print(f"  放下时停住：{reason}")
                            break
                        rig.move(qt, keys)
                    rig.set_hand("open")
                    closed = False
            if not closed and confirm(args, keys, "\n  抬到起步高度并退回起始位？回车=是  q=停在这里 > ") == "q":
                raise rc.StopRequested("操作者选择停在这里")
            if not closed:
                print("  抬到起步高度 → 回起步点 → 沿示范退回起始位 ...")
                qn = rig.joints()
                pn = kin.fk(qn)[:3, 3] * 1000
                p_st = np.asarray(args.start, float)
                p_up = np.array([pn[0], pn[1], max(pn[2], p_st[2])])
                while np.linalg.norm(p_up - pn) > 0.5:
                    nxt = pn + np.clip(p_up - pn, -20, 20)
                    qt, reason = kin.solve(qn, nxt, R_g, floor)
                    if qt is None:
                        raise RuntimeError(f"退回时抬不上去：{reason}")
                    rig.move(qt, keys)
                    qn, pn = qt, nxt
                p_s = p_st
                while np.linalg.norm(p_s - pn) > 0.5:
                    nxt = pn + np.clip(p_s - pn, -20, 20)
                    qt, reason = kin.solve(qn, nxt, R_g, floor)
                    if qt is None:
                        raise RuntimeError(f"退回时平移不过去：{reason}")
                    rig.move(qt, keys)
                    qn, pn = qt, nxt
                for qq in reversed(q_rot[:-1]):
                    rig.move(qq, keys)
                rig.move(q[lead[-1]], keys)
                for i in reversed(lead[:-1]):
                    rig.move(q[i], keys)
                print("  已回到起始位。")
    except rc.StopRequested as e:
        print(f"\n停下：{e}。臂停在原地，手保持 {rig.hand_state}。")
    except Exception as e:  # noqa: BLE001
        print(f"\n出错停下：{type(e).__name__}: {e}。臂停在原地，手保持 {rig.hand_state}。")
    finally:
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        n_move = sum(1 for r in log if r.get("user") == "enter")
        n_skip = sum(1 for r in log if r.get("user") == "s")
        print(f"共 {len(log)} 步：执行 {n_move}、人工跳过 {n_skip}、自动重问 "
              f"{sum(1 for r in log if r.get('user') == 'auto-skip')} → {out}")
        cv2.destroyAllWindows()
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
    ap.add_argument("--floor", type=float, default=None, help="法兰最低 z（mm），默认示范抓取高度 −10")
    ap.add_argument("--start", type=float, nargs=3, default=(380.0, -300.0, 330.0),
                    help="闭环起步的法兰位置 mm（抓取朝向）；默认在白板右半边，j7 余量大")
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_loop")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-empty", action="store_true", help=argparse.SUPPRESS)   # mock 时把合手当成空夹
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
