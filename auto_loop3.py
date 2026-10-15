#!/usr/bin/env python3
"""能力测试：不分阶段、完全不用示教、每步一道题，看模型自己能做到哪些。

输入（每步）：俯视彩色、俯视深度、腕部彩色、腕部深度、侧面彩色 + 一张旋转参考图（同一只手六种转法转完后腕部看到的样子）
选项（一道题，正序倒序各问一次）：
  四个水平方向 × 大步 25mm / 小步 6mm、下降 15mm（指尖碰到东西就自动停）、上升 20mm、
  六种旋转（绕法兰 x/y/z 正反各 10°，绕两指之间那点）、合手、张手、完成、放弃
不替模型做决定：碰到之后合不合手、夹住之后抬不抬、什么时候算完成，全由模型选；物理结果（碰到了、手指读数说夹没夹住）
写进下一步的文字状态里。
不用示教：起始朝向；俯视图只在角上画方向示意（北 = 画面上方，由不用示教的几次真跑确认）；腕部方向用 calib_start 的 J；
腕部瞄准点 = 空中合手照片；深度走纯 V4L2（depth_v4l2.py）。
安全（物理的）：逆解、软限位、单步 ≤10°、法兰最低线随转角放宽、下降碰到就停、连续 3 步被拒就停、接口不可用正常收尾；
结束时手里有东西就放回（降到碰到再张手），再回起始位。

    python auto_loop3.py --dry-run
    python auto_loop3.py --mock
    python auto_loop3.py --auto
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
import auto_loop2 as L  # noqa: E402
import depth_v4l2 as D  # noqa: E402

LIFT_Z = L.LIFT_Z
BIG, SMALL, DOWN, UP = 25.0, 6.0, 15.0, 20.0
MAX_STEPS = 120
DIRS = {"north": (1, 0, 0), "south": (-1, 0, 0), "west": (0, 1, 0), "east": (0, -1, 0)}
TOP_COMPASS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}     # 俯视图里的方向（真跑确认）
COL = {"north": (0, 0, 255), "east": (0, 200, 255), "south": (255, 255, 0), "west": (255, 0, 255)}

CONTEXT = (
    "You control a robot arm with a dexterous hand. Only the THUMB and the INDEX finger move: the hand grasps by "
    "pinching an object between the thumb tip and the index fingertip; the other three fingers stay straight. "
    "Task: pick up the blue sleeve (a short blue cylinder) standing on the white board, lift it about 8 cm, then choose done.\n"
    "Images:\n"
    "1. TOP camera (fixed, looking straight down at the table). The compass in the corner shows which way each move "
    "direction goes in this image (north = up, south = down, west = left, east = right). The robot arm reaches in "
    "from the bottom-left.\n"
    "2. TOP depth (same camera position, not pixel-aligned with image 1): colour shows distance from the camera, "
    "red = near, blue = far, black = no data. Things standing on the board look warmer than the board.\n"
    "3. HAND camera (fixed on the hand, looking down past the fingers). The dark finger at the LEFT edge is the index "
    "finger, the dark finger at the RIGHT edge is the thumb. The small green dot in the green ring is where the two "
    "fingertips meet when the hand closes: an object must be under that dot to be pinched. The four coloured arrows "
    "from the ring show how each move direction shifts the hand in this image.\n"
    "4. HAND depth (5 cm red ... 45 cm blue from the hand camera; black = no data).\n"
    "5. SIDE camera at table level: shows the fingertips and the sleeve from the side.\n"
    "6. Rotation reference: panel NOW is the hand-camera view in the hand's normal orientation; each lettered panel "
    "shows the same view after that rotation of the hand (photographed earlier).\n"
    "Moving down stops automatically as soon as a fingertip touches something.")
Q = "Which single action should the robot take next?"


# ── 画图 ────────────────────────────────────────────────────────────────
def top_compass(top):
    img = top.copy()
    o = np.array([48, 52])
    cv2.rectangle(img, (4, 4), (96, 100), (255, 255, 255), -1)
    for d, v in TOP_COMPASS.items():
        b = o + np.array(v) * 34
        cv2.arrowedLine(img, tuple(int(x) for x in o), tuple(int(x) for x in b), COL[d], 3, tipLength=0.35)
        t = o + np.array(v) * 44 - [5, -5]
        cv2.putText(img, d[0].upper(), (int(t[0]), int(t[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


def wrist_compass(wrist, J):
    img = L.ring(wrist.copy())
    h, w = img.shape[:2]
    o = np.asarray(L.PINCH_PX, float)
    for d, v3 in DIRS.items():
        v = -(J @ np.array(v3[:2], float))
        v /= np.linalg.norm(v)
        a = np.clip(o + v * (L.RING_R + 8), [10, 10], [w - 10, h - 10])
        b = np.clip(o + v * (L.RING_R + 70), [10, 10], [w - 10, h - 10])
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (0, 0, 0), 5, tipLength=0.3)
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), COL[d], 3, tipLength=0.3)
        c = np.clip(o + v * (L.RING_R + 40) + np.array([v[1], -v[0]]) * 17, [11, 11], [w - 11, h - 11]).astype(int)
        cv2.circle(img, tuple(c), 10, (255, 255, 255), -1)
        cv2.putText(img, d[0].upper(), (int(c[0]) - 6, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


def options(rot_letters, allowed):
    ch = []
    for d in DIRS:
        ch.append((f"{d}_big", f"Move the hand {d} by {BIG:.0f} mm."))
        ch.append((f"{d}_small", f"Move the hand {d} by {SMALL:.0f} mm."))
    ch += [("lower", f"Move the hand down by up to {DOWN:.0f} mm (stops if a fingertip touches something)."),
           ("raise", f"Move the hand up by {UP:.0f} mm.")]
    ch += [(f"rotate_{k}", f"Rotate the hand as in panel {k} of image 6 (about 10 degrees).")
           for k, v in rot_letters.items() if v in allowed]
    ch += [("close", "Close the thumb and index finger to pinch."), ("open", "Open the thumb and index finger."),
           ("done", "The sleeve is held in the hand and lifted off the board: the task is complete."),
           ("give_up", "The blue sleeve cannot be found anywhere.")]
    return ch


def make_decider(args, out):
    if not args.mock:
        return L.make_decider(args, out)
    rnd = random.Random(3)
    plan = (["east_big"] * 5 + ["south_big", "east_small", "rotate_A"] + ["lower"] * 6 + ["close", "open", "raise",
            "rotate_B"] + ["lower"] * 5 + ["close"] + ["raise"] * 5 + ["done"])
    k = {"i": 0}

    def fake(req, kind):
        values = [c["value"] for c in req["questions"][0]["choices"]]
        v = plan[min(k["i"], len(plan) - 1)]
        k["i"] += 1
        if v not in values:
            v = values[0]
        b = v if rnd.random() > 0.2 else values[-3]
        return {"answers": {"a": {"choice": v}, "b": {"choice": b}}, "ms": 1.0, "input_tokens": 0}
    return fake


# ── 主流程 ───────────────────────────────────────────────────────────────
def run(args):
    from experiments.robot.ar5.config import AR5Config
    kin = CL.Kin()
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    T0 = kin.fk(q_home)
    R0, p_home = T0[:3, :3], T0[:3, 3] * 1000
    lift_path = [(np.array([p_home[0], p_home[1], z]), R0)
                 for z in list(np.arange(p_home[2] + 10, LIFT_Z - 0.01, 10)) + [LIFT_Z]]
    print(f"模型 {E.MODEL}；每步一道题、{len(options(dict(zip('ABCDEF', L.ROTS)), list(L.ROTS)))} 个选项、6 张图；最多 {MAX_STEPS} 步")
    if args.dry_run:
        print("离线演算：只检查参数，没连硬件、没调接口。")
        return 0
    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"记录 → {out}")
    if not args.mock and not args.auto:
        if input("  全自动能力测试：套筒立在白板上、臂在起始位。确认后输入 yes 回车: ").strip().lower() != "yes":
            return 0

    rc.HAND_OPEN, rc.OTHERS_OPEN = L.OPEN_IDX_THUMB, L.OTHERS_OPEN
    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    dec = make_decider(args, out)
    rng = np.random.default_rng(int(time.time()))
    depth = {}
    log, t0, step, result = [], time.time(), 0, "未完成"
    st = {"hand": "open", "held": False, "last": "start", "touch_z": None, "rejected": 0, "no_rot": set(),
          "closes": 0, "empty_closes": 0, "grasp_z": None}
    pool = []
    try:
        if not args.mock:
            depth["top"] = D.DepthStream(D.DEPTH_TOP, 1.0).start()
            depth["wrist"] = D.DepthStream(D.DEPTH_WRIST, 0.1).start()
        rig.connect()
        if np.degrees(np.abs(rig.joints() - q_home)).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在起始位。先用 rescue_home.py 回位")
        with rc.Keys(mock=args.mock) as keys:
            arm = L.Arm(rig, kin, keys)
            rig.set_hand("open")
            lift_seq, why = arm.plan(lift_path, floor=-1e9)
            if lift_seq is None:
                raise RuntimeError(f"起步走不了：{why}")
            arm.run(lift_seq)
            cmd = [lift_path[-1][0].copy(), R0.copy()]

            def plan_go(path, recover=False):
                floor = L.hard_min(path[-1][1], R0)
                seq, why = arm.plan(path, floor=floor, start=cmd)
                if seq is None and recover:                     # 够不着：绕穿过两指之间那点的竖直轴转一下让开
                    p, R = cmd
                    c = p + R @ R0.T @ L.PIVOT0
                    for yaw in (15, -15, 30, -30):
                        Rz = CL.rot(np.array([0, 0, 1.0]), np.radians(yaw))
                        p_y, R_y = c + Rz @ (p - c), Rz @ R
                        p_t = p_y + (np.asarray(path[-1][0], float) - p)
                        seq2, _ = arm.plan([(p_y, R_y), (p_t, R_y)], floor=L.hard_min(R_y, R0), start=cmd)
                        if seq2 is not None:
                            seq, path = seq2, [(p_y, R_y), (p_t, R_y)]
                            break
                if seq is None:
                    return False, why
                arm.run(seq)
                cmd[0], cmd[1] = np.asarray(path[-1][0], float), path[-1][1]
                return True, ""

            def descend(dz):
                base = L.forces(rig.hand, 7)
                p, R = cmd[0].copy(), cmd[1]
                target = p[2] - dz
                while p[2] - L.TOUCH_STEP >= target - 1e-6:
                    ok, why = plan_go([(p + [0, 0, -L.TOUCH_STEP], R)])
                    if not ok:
                        return f"could not move down further (arm limit) at {p[2]:.0f} mm"
                    p = cmd[0].copy()
                    f = L.forces(rig.hand)
                    if args.mock and p[2] <= 285:
                        f = base + [0, 20, 0, 0, 0, 0]
                    d = f - base
                    k = int(np.argmax(np.abs(d)))
                    if abs(d[k]) > L.TOUCH_TH:
                        plan_go([(p + [0, 0, L.TOUCH_STEP], R)])
                        st["touch_z"] = p[2]
                        return "a fingertip touched something and the hand stopped (moved back up 2 mm)"
                return f"moved down {dz:.0f} mm without touching anything"

            try:
                while step < MAX_STEPS:
                    step += 1
                    keys.poll()
                    q, p_meas, _ = arm.now()
                    p, R = cmd[0].copy(), cmd[1]
                    J = L.J_of(R, R0)
                    rot_deg = np.degrees(CL.axis_angle(R0.T @ R)[1])
                    fr = CL.grab(rig)
                    top, wrist, side = fr.get("cam_high"), fr.get("cam_front"), fr.get("cam_side")
                    dt = depth["top"].get_mm() if "top" in depth else None
                    dw = depth["wrist"].get_mm() if "wrist" in depth else None
                    dt = dt if dt is not None else np.zeros((480, 640), np.float32)     # 深度流卡住：给空图，不报错
                    dw = dw if dw is not None else np.zeros((480, 640), np.float32)
                    imgs = {"top": top_compass(top), "top_depth": D.colorize(dt, 350, 750, "top"),
                            "wrist": wrist_compass(wrist, J), "wrist_depth": D.colorize(dw, 50, 450, "hand"),
                            "side": side}
                    allowed = [r for r in L.ROTS if r not in st["no_rot"]]
                    rot_letters = dict(zip("ABCDEF", rng.permutation(list(L.ROTS))))
                    imgs["rot_ref"] = L.ref_sheet(rot_letters, allowed)
                    for k, im in imgs.items():
                        cv2.imwrite(str(out / f"{step:03d}_{k}.jpg"), im)
                    np.save(out / f"{step:03d}_depth_wrist.npy", dw.astype(np.float16))
                    state = (f"Current state: the hand is {st['hand']}"
                             + (" and its fingers are blocked by an object (something is pinched)" if st["held"] else "")
                             + f". The last action was: {st['last']}.")
                    text = CONTEXT + "\n" + state
                    ims = [("Image 1 (top camera):", imgs["top"]), ("Image 2 (top depth):", imgs["top_depth"]),
                           ("Image 3 (hand camera):", imgs["wrist"]), ("Image 4 (hand depth):", imgs["wrist_depth"]),
                           ("Image 5 (side camera):", imgs["side"]), ("Image 6 (rotation reference):", imgs["rot_ref"])]
                    req = {"model": E.MODEL, "input": A.msg(text, *ims), "questions": A.two_orders("x", Q, options(rot_letters, allowed))}
                    res = dec(req, "all")
                    sem = lambda x: ("rotate_" + rot_letters[x[-1]]) if x.startswith("rotate_") else x  # noqa: E731
                    a, b = sem(res["answers"]["a"]["choice"]), sem(res["answers"]["b"]["choice"])
                    choice = A.vote(pool, a, b)
                    rec = {"step": step, "t": round(time.time() - t0, 1), "flange": p.round(1).tolist(),
                           "flange_meas": p_meas.round(1).tolist(), "rot_deg": round(rot_deg, 1), "hand": st["hand"],
                           "held": st["held"], "answer": [a, b], "state_text": state,
                           "conf": [res["answers"][o].get("confidence") for o in ("a", "b")]}
                    note = ""
                    if choice is None:
                        action = "reask"
                    else:
                        action = choice
                        pool.clear()
                        if action.endswith("_big") or action.endswith("_small"):
                            d, size = action.split("_")
                            mm = BIG if size == "big" else SMALL
                            ok, why = plan_go([(p + np.array(DIRS[d]) * mm, R)], recover=True)
                            note = f"moved {d} {mm:.0f} mm" if ok else f"could not move {d} (arm limit: {why})"
                        elif action == "lower":
                            note = descend(DOWN)
                        elif action == "raise":
                            ok, why = plan_go([(p + [0, 0, UP], R)], recover=True)
                            note = f"moved up {UP:.0f} mm" if ok else f"could not move up (arm limit)"
                        elif action.startswith("rotate_"):
                            rname = action.split("_")[1]
                            ax, sg = L.ROTS[rname]
                            R1 = R @ CL.rot(np.eye(3)[ax], np.radians(sg * L.ROT_DEG))
                            c = p + R @ R0.T @ L.PIVOT0
                            if np.degrees(CL.axis_angle(R0.T @ R1)[1]) > L.MAX_ROT_DEG:
                                ok, why = False, "more than 45 degrees from the start orientation"
                            else:
                                ok, why = plan_go([(c + R1 @ R.T @ (p - c), R1)])
                            if not ok:
                                st["no_rot"].add(rname)
                            note = f"rotated ({rname} 10°)" if ok else "could not rotate that way (arm limit)"
                        elif action == "close":
                            ang = rig.set_hand("close")
                            held, ex_i, ex_t = CL.finger_held(ang)
                            if args.mock:
                                held = st["closes"] >= 1
                            st["closes"] += 1
                            st["hand"], st["held"] = "closed", bool(held)
                            st["empty_closes"] += not held
                            if held:
                                st["grasp_z"] = p[2]
                            rec["fingers"] = [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])]
                            note = ("closed: the fingers were blocked by an object (something is pinched)" if held
                                    else "closed: the fingers closed fully on nothing (empty)")
                        elif action == "open":
                            rig.set_hand("open")
                            st["hand"], st["held"] = "open", False
                            note = "opened the hand"
                        elif action in ("done", "give_up"):
                            note = action
                        if note.startswith("could not"):
                            st["rejected"] += 1
                        else:
                            st["rejected"] = 0
                        st["last"] = f"{action} -> {note}"
                        if st["held"] and action == "raise":                   # 抬的时候确认还夹着
                            held_now = CL.finger_held(rig.hand.get_angles())[0] or args.mock
                            if not held_now:
                                st["held"] = False
                                st["last"] += " (the fingers are no longer blocked: the object slipped out)"
                    rec.update(action=action, note=note)
                    log.append(rec)
                    print(f"step {step:3d} t={rec['t']:5.1f}s 法兰({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) 转{rot_deg:4.1f}° "
                          f"手{'合' if st['hand'] == 'closed' else '张'}{'·夹着' if st['held'] else ''} -> {action} [{a}/{b}] | {note}",
                          flush=True)
                    if action == "done":
                        lifted = st["grasp_z"] is not None and p[2] - st["grasp_z"]
                        ok_done = st["held"] and lifted and lifted >= 50
                        result = (f"模型说完成：{'成功（夹着、抬高 %.0fmm）' % lifted if ok_done else '但实际没完成（夹着=%s、抬高=%s）' % (st['held'], lifted)}")
                        break
                    if action == "give_up":
                        result = "模型放弃：说找不到套筒"
                        break
                    if st["rejected"] >= 3:
                        result = f"停止：连续 3 步动不了（{note}）"
                        break
                else:
                    result = f"停止：到了 {MAX_STEPS} 步上限"
            except L.ApiDown as e:
                result = f"停止：决策接口不可用（{e}）"
            print(f"\n结果：{result}（{step} 步，{time.time() - t0:.0f}s）", flush=True)
            # 收尾：手里有东西就放回（降到碰到再张手）→ 抬升高度从高往低试 → 回白板中间上方 → 转回起始朝向 → 原路降回
            if st["hand"] == "closed":
                if st["held"]:
                    print("放回套筒 ...", flush=True)
                    descend(120)
                rig.set_hand("open")
            print("回起始位 ...", flush=True)
            p, R = cmd
            back, why = None, ""
            for z in sorted({max(p[2], LIFT_Z + 20), 345.0, 330.0, 315.0, round(p[2] + 30, 1), round(float(p[2]), 1)}, reverse=True):
                if z < p[2] + (20 if p[2] < 320 else 0):     # 10-14：已在 340 高处时 360/370 够不着，允许就在当前高度出发
                    continue
                zc = max(z, LIFT_Z)
                path_back = [(np.array([p[0], p[1], z]), R), (np.array([450.0, 0.0, zc]), R),
                             (np.array([450.0, 0.0, zc]), R0), (np.array([p_home[0], p_home[1], LIFT_Z]), R0)]
                back, why = arm.plan(path_back, floor=-1e9, start=cmd)
                if back is not None:
                    break
            if back is None:
                raise RuntimeError(f"回程走不了：{why}。停在这里，用 rescue_home.py 回位")
            arm.run(back)
            q_now, q_top = rig.joints(), lift_seq[-1]
            n = max(1, int(np.ceil(np.degrees(np.abs(q_top - q_now)).max() / 3.0)))
            for k in range(1, n + 1):
                qk = q_now + (q_top - q_now) * k / n
                if kin.fk(qk)[2, 3] * 1000 < LIFT_Z - 40:
                    raise RuntimeError("关节过渡途中法兰会太低，停在这里，用 rescue_home.py 回位")
                rig.move(qk, keys)
            for qq in reversed(lift_seq[:-1]):
                rig.move(qq, keys)
            rig.move(q_home, keys)
            print("已回到起始位。", flush=True)
    except rc.StopRequested as e:
        result = f"急停：{e}"
    except Exception as e:  # noqa: BLE001
        result = f"出错：{type(e).__name__}: {e}"
    finally:
        # 10-14：中断（SIGINT）时这段原来不跑 → 日志没存、深度流和臂会话没关。放进 finally
        if result == "未完成":
            result = "中断"
        summary = {"result": result, "steps": step, "seconds": round(time.time() - t0, 1), "model": E.MODEL,
                   "closes": st["closes"], "empty_closes": st["empty_closes"], "log": log}
        (out / "log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        for s_ in depth.values():
            s_.stop()
        rig.close()
        print(f"记录 → {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_auto3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--auto", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
