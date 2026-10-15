#!/usr/bin/env python3
"""RobotDecision 全自动闭环：不用示教路径起步、不要人每步确认，三路相机分段决策。

起步（固定动作，不问模型）
  起始位竖直抬到 LIFT_Z（起始位在白板侧面偏低，直接平移会撞板边）→ 位置和朝向一起插值到 --start
  （白板右半边上方、抓取朝向；那里 j7 余量够）。全程逆解逐步检查，单步关节 ≤ MAX_JUMP_DEG。

每一步（全由 Decisions 选）
  ① 位置题（腕部图 + 环 + 中心点）：套筒 not_visible / outside / inside —— 决定用哪一段
  ② 粗定位 not_visible：只看俯视图（箭头）→ 选箭头，移 STEP_FAR；或 target_not_visible → 停
     进环前 outside：俯视 + 腕部（两张图同字母箭头）→ 选箭头，移 STEP_MID
     环内 inside：腕部（环 + 中心点 + 箭头）+ 侧面 → 选箭头（把套筒中心对到中心点）移 STEP_RING，
                 或 lower（对准了、侧面里指尖还在套筒顶上方）降 STEP_DOWN，或 close
  选择题都正序、倒序各问一次；不一致不动、重拍重问；同一位姿 3 轮都不一致就取所有答案的多数票
合手
  合手后看手指读数（被挡住 = 夹住）。空夹 → 张手、降 5mm 再试，最多 MAX_ATTEMPTS 次；
  模型要降但已到法兰下限 → 在下限试合。夹住 → 规则抬 LIFT_STEPS×1cm，每步看手指；够了 = 完成
  10-13 离线验证：侧面图判不了指尖高度（三种问法都不随高度变，二选一题跟着选项顺序走）。
  所以实际合手多半是「降到下限 → 试合 → 手指读数确认」，侧面图只是给模型参考。
安全（物理的，不是画框）：逆解可达、软限位余量、单步关节 ≤10°、法兰 ≥ --floor；
  连续 3 步被拒就停并报原因；运行中按 q 急停

    python auto_loop.py --dry-run
    python auto_loop.py --mock
    python auto_loop.py              # 真跑（开始前要输入 yes）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_capture as rc  # noqa: E402
import eval_decisions as E  # noqa: E402
import closed_loop as CL  # noqa: E402

LIFT_Z = 340.0
STEP_FAR, STEP_MID, STEP_RING, STEP_DOWN, RAISE = 20.0, 15.0, 5.0, 8.0, 10.0
LIFT_STEPS, MAX_ATTEMPTS, MAX_STEPS, MAX_REASK = 8, 4, 150, 3
DIR_BASE = CL.DIR_BASE
LETTER_DIRS = ["north", "east", "south", "west"]


# ── 画图 ─────────────────────────────────────────────────────────────────
def ring(img, dot=True):
    (cu, cv_), rad = E.WRIST_CIRCLE
    cv2.circle(img, (cu, cv_), rad, (0, 0, 0), 6)
    cv2.circle(img, (cu, cv_), rad, (0, 255, 0), 3)
    if dot:
        cv2.circle(img, (cu, cv_), 7, (0, 0, 0), -1)
        cv2.circle(img, (cu, cv_), 5, (0, 255, 0), -1)
    return img


def wrist_with_arrows(wrist, letters):
    img = ring(wrist.copy())
    (cu, cv_), _ = E.WRIST_CIRCLE
    E.draw_wrist_arrows(img, (cu, cv_), letters, False)
    return img


def top_with_arrows(top, flange_xy, letters):
    img = top.copy()
    E.draw_arrows(img, E.TOP_MAP(flange_xy), letters)
    return img


def msg(text, *images):
    content = [{"type": "input_text", "text": text}]
    for label, im in images:
        content += [{"type": "input_text", "text": label},
                    {"type": "input_image", "image_url": CL.url(im), "detail": "auto"}]
    return [{"role": "user", "content": content}]


def two_orders(name, instr, choices):
    ch = [{"value": v, "description": d} for v, d in choices]
    return [{"type": "choice", "name": "a", "instructions": instr, "choices": ch},
            {"type": "choice", "name": "b", "instructions": instr, "choices": ch[::-1]}]


# ── 四道题 ───────────────────────────────────────────────────────────────
PHASE_TEXT = ("This image is from a camera fixed on a robot hand, looking down at a table. The GREEN RING marks the "
              "zone where an object must be for the fingers to grasp it; the small green dot is the ring's center.")
PHASE_Q = "Where is the blue sleeve in this image?"
PHASE_CH = [("not_visible", "No blue sleeve is visible in the image."),
            ("outside", "A blue sleeve is visible, but its center is outside the green ring."),
            ("inside", "The center of the blue sleeve is inside the green ring.")]

FAR_TEXT = ("This image is from a fixed camera looking straight down at a work table (a white board on a green mat). "
            "A robot arm reaches in from the bottom-left. Four lettered arrows start at the robot hand; each shows "
            "one horizontal direction the hand can move. Task: bring the hand over the blue sleeve.")
FAR_Q = "Which arrow moves the hand toward the blue sleeve?"

MID_TEXT = ("Image 1 is from a fixed camera looking straight down at a work table; the robot arm reaches in from "
            "the bottom-left. Image 2 is from a camera fixed on the robot hand, looking down; the green ring marks "
            "where the object must be for the fingers to grasp it. The same four lettered arrows are drawn in both "
            "images: arrow X in image 2 is the same move as arrow X in image 1 (they point differently because the "
            "hand camera is turned). Task: bring the blue sleeve into the green ring.")
MID_Q = "Which arrow moves the hand so that the blue sleeve comes toward the green ring?"

RING_TEXT = ("Image 1 is from a camera fixed on a robot hand, looking down. The green ring is the grasping zone and "
             "the small green dot is its center. Four lettered arrows show four short horizontal moves of the hand. "
             "Image 2 is from a camera at table level that shows the robot's fingertips and the blue sleeve from "
             "the side. Goal: put the center of the blue sleeve on the green dot, then bring the fingertips down "
             "to the sleeve and close the hand.")
RING_Q = ("What should the hand do next? If the center of the blue sleeve is not on the green dot in image 1, choose "
          "the arrow pointing from the dot toward the sleeve's center. If it is on the dot and the fingertips in "
          "image 2 are still clearly above the top of the sleeve, choose lower. If it is on the dot and the "
          "fingertips have come down to the top rim of the sleeve, choose close.")


def ask_phase(dec, wrist):
    img = ring(wrist.copy())
    req = {"model": E.MODEL, "input": msg(PHASE_TEXT, ("", img)), "questions": two_orders("p", PHASE_Q, PHASE_CH)}
    return dec(req, "phase"), img


def ask_far(dec, top, flange_xy, letters):
    img = top_with_arrows(top, flange_xy, letters)
    ch = [(f"arrow_{k}", f"Arrow {k}.") for k in "ABCD"] + [("target_not_visible", "No blue sleeve is visible.")]
    req = {"model": E.MODEL, "input": msg(FAR_TEXT, ("", img)), "questions": two_orders("f", FAR_Q, ch)}
    return dec(req, "far"), img


def ask_mid(dec, top, wrist, flange_xy, letters):
    t = top_with_arrows(top, flange_xy, letters)
    w = wrist_with_arrows(wrist, letters)
    ch = [(f"arrow_{k}", f"Arrow {k}.") for k in "ABCD"]
    req = {"model": E.MODEL, "input": msg(MID_TEXT, ("Image 1 (top camera):", t), ("Image 2 (hand camera):", w)),
           "questions": two_orders("m", MID_Q, ch)}
    return dec(req, "mid"), t, w


def ask_ring(dec, wrist, side, letters):
    w = wrist_with_arrows(wrist, letters)
    ch = [(f"arrow_{k}", f"Move along arrow {k}.") for k in "ABCD"]
    ch += [("lower", "Lower the hand a short step."), ("close", "Close the thumb and index finger to grasp.")]
    req = {"model": E.MODEL, "input": msg(RING_TEXT, ("Image 1 (hand camera):", w), ("Image 2 (side camera):", side)),
           "questions": two_orders("r", RING_Q, ch)}
    return dec(req, "ring"), w


def vote(pool, a, b):
    """正序、倒序两次一致 → 用它。不一致 → 记进 pool、返回 None（重拍重问）；
    同一位姿攒满 MAX_REASK 轮不一致 → 取所有答案里票数最多的（平票取先出现的）。"""
    if a == b:
        return a
    pool += [a, b]
    if len(pool) < 2 * MAX_REASK:
        return None
    return Counter(pool).most_common(1)[0][0]


def make_decider(args, log_dir):
    if not args.mock:
        return CL.make_decider(args, log_dir)
    rnd = random.Random(1)
    script = {"phase": ["not_visible", "not_visible", "outside", "outside", "inside", "inside", "inside", "inside",
                        "inside", "inside", "inside", "inside"],
              "far": ["arrow_A", "arrow_B"], "mid": ["arrow_C", "arrow_A"],
              "ring": ["arrow_B", "close"] + ["lower"] * 30}     # 先在半空合一次（空夹），再一路降到下限
    pos = {k: 0 for k in script}

    def fake(req, kind):
        seq = script[kind]
        v = seq[min(pos[kind], len(seq) - 1)]
        pos[kind] += 1
        b = v if rnd.random() > 0.3 else ("outside" if kind == "phase" else "arrow_D")   # 不一致，测重问和多数票
        return {"answers": {"a": {"choice": v}, "b": {"choice": b}}, "ms": 1.0, "input_tokens": 0}
    return fake


# ── 运动 ─────────────────────────────────────────────────────────────────
def takeoff_plan(kin, q0, R_g, start):
    """起始位竖直抬到 LIFT_Z，再位置 + 朝向一起插值到 start。返回关节序列（不含 q0）。"""
    T0 = kin.fk(q0)
    p0, R0 = T0[:3, 3] * 1000, T0[:3, :3]
    seq, qs = [], q0
    for z in np.arange(p0[2] + 10, LIFT_Z + 0.01, 10):
        qs, why = kin.solve(qs, np.array([p0[0], p0[1], z]), R0, -1e9)
        if qs is None:
            return None, f"竖直抬到 z{z:.0f} 失败：{why}"
        seq.append(qs)
    pl = np.array([p0[0], p0[1], LIFT_Z])
    ps = np.asarray(start, float)
    ax, ang = CL.axis_angle(R0.T @ R_g)
    for s in range(1, 41):
        f = s / 40
        qs, why = kin.solve(qs, pl + (ps - pl) * f, R0 @ CL.rot(ax, ang * f), -1e9)
        if qs is None:
            return None, f"过渡第 {s}/40 步失败：{why}"
        seq.append(qs)
    return seq, f"竖直抬到 z{LIFT_Z:.0f}，再转 {np.degrees(ang):.0f}°、移到 {tuple(int(v) for v in ps)}，共 {len(seq)} 步"


def run(args):
    kin = CL.Kin()
    plan = rc.make_plan(args)                        # 只用来取抓取朝向 R_g 和示范抓取高度（法兰下限的参考）
    g = plan["i_grasp"]
    R_g = plan["T"][g][:3, :3]
    floor = plan["P"][g, 2] - 10.0 if args.floor is None else args.floor
    from atom_robot.ar5.config import AR5Config
    q_home = np.asarray(AR5Config(mock=True).arm.home_joints, float)
    seq, why = takeoff_plan(kin, q_home, R_g, args.start)
    if seq is None:
        print(f"✗ 起步解不出：{why}")
        return 1
    jumps = np.degrees(np.abs(np.diff(np.vstack([q_home, *seq]), axis=0))).max(1)
    print(f"起步：{why}；单步关节最多 {jumps.max():.1f}°")
    print(f"法兰下限 {floor:.0f}mm；步长 粗 {STEP_FAR:.0f} / 进环前 {STEP_MID:.0f} / 环内 {STEP_RING:.0f} / "
          f"降 {STEP_DOWN:.0f} / 抬 {RAISE:.0f} mm；最多 {MAX_STEPS} 步、合手最多试 {MAX_ATTEMPTS} 次")
    if args.dry_run:
        print("--dry-run：没连硬件、没调 API。")
        return 0

    out = Path(args.out).expanduser() / time.strftime("%Y%m%d_%H%M%S" + ("_mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"记录 → {out}")
    if not args.mock:
        print("=" * 66)
        print("  ⚠ 全自动：臂会自己抬起、过去、对准、合手、抬起，不再每步等回车。")
        print("    开始前：套筒放在白板右半边，臂在起始位。随时按 q 急停（走完当前一小步就停）。")
        print("=" * 66)
        if input("  确认后输入 yes 回车: ").strip().lower() != "yes":
            print("  取消。")
            return 0

    rig = rc.Rig(argparse.Namespace(mock=args.mock, speed=args.speed), out)
    dec = make_decider(args, out)
    rng = np.random.default_rng(int(time.time()))
    log, t_start = [], time.time()
    closed, lifted, attempts, step, result = False, 0, 0, 0, "未完成"
    phase_pool, act_pool, last_phase = [], [], None      # 同一位姿下攒的不一致答案；动过就清空
    rejected = 0                                         # 连续被安全检查拒掉的步数
    try:
        rig.connect()
        if np.degrees(np.abs(rig.joints() - q_home)).max() > 3.0 and not args.mock:
            raise RuntimeError("臂不在起始位。先回位（灵巧手遥操/VR设备/home_arm.py）再跑")
        with rc.Keys(mock=args.mock) as keys:
            rig.set_hand("open")
            print("\n起步 ...")
            for qq in seq:
                rig.move(qq, keys)

            def try_close(rec):
                nonlocal closed, attempts
                attempts += 1
                ang = rig.set_hand("close")
                held, ex_i, ex_t = CL.finger_held(ang)
                if args.mock:
                    held = attempts >= 2                    # mock：第一次空夹，第二次夹住
                rec["grasp"] = {"try": attempts, "index": int(ang[rc.INDEX]), "thumb": int(ang[rc.THUMB_BEND]),
                                "held": bool(held)}
                print(f"  合手第 {attempts} 次：食指 {ang[rc.INDEX]} / 拇指 {ang[rc.THUMB_BEND]}"
                      f"（+{ex_i}/+{ex_t}）→ {'夹住了' if held else '空夹'}")
                if held:
                    closed = True
                    return
                rig.set_hand("open")
                qn = rig.joints()
                pn = kin.fk(qn)[:3, 3] * 1000
                dz = max(pn[2] - 5.0, floor) - pn[2]
                qt = kin.solve(qn, pn + [0, 0, dz], R_g, floor)[0] if dz < -1.0 else None
                if qt is not None:
                    rig.move(qt, keys)
                    print(f"  空夹：张手、降 {-dz:.0f}mm 再找")
                else:
                    print("  空夹：张手（已在下限，不再降）")

            while step < MAX_STEPS:
                step += 1
                keys.poll()
                qn = rig.joints()
                pn = kin.fk(qn)[:3, 3] * 1000
                fr = CL.grab(rig)
                top, wrist, side = fr.get("cam_high"), fr.get("cam_front"), fr.get("cam_side")
                for k, im in (("top", top), ("wrist", wrist), ("side", side)):
                    if im is not None:
                        cv2.imwrite(str(out / f"{step:03d}_{k}.jpg"), im)
                rec = {"step": step, "t": round(time.time() - t_start, 1), "flange": pn.round(1).tolist()}
                top_ann, wr_ann = top, wrist
                action, delta, note = None, None, ""

                if closed:
                    ang = rig.hand.get_angles()
                    held, ex_i, ex_t = CL.finger_held(ang)
                    held = held or args.mock
                    rec["fingers"] = [int(ang[rc.INDEX]), int(ang[rc.THUMB_BEND])]
                    if not held:
                        action, note = "stop", f"抬起途中手指不再被挡住（{ang[rc.INDEX]}/{ang[rc.THUMB_BEND]}）= 掉了"
                    elif lifted >= LIFT_STEPS:
                        action, note = "done", f"已抬 {lifted * RAISE:.0f}mm，手指仍被挡住"
                    else:
                        action, delta, note = "raise", (0, 0, RAISE), "夹着，按规则抬"
                else:
                    res, wr_ann = ask_phase(dec, wrist)
                    pa, pb = res["answers"]["a"]["choice"], res["answers"]["b"]["choice"]
                    rec["phase"] = [pa, pb]
                    phase = vote(phase_pool, pa, pb)
                    if phase is None:
                        action, note = "reask", f"位置题两次不一致 {pa}/{pb}"
                    else:
                        if pa != pb:
                            note = f"位置题 {MAX_REASK} 轮不一致，多数票 {phase}；"
                        if phase != last_phase:
                            act_pool.clear()
                        last_phase = phase
                        letters = dict(zip("ABCD", rng.permutation(LETTER_DIRS)))
                        if phase == "not_visible":
                            res, top_ann = ask_far(dec, top, pn[:2], letters)
                            stage, step_mm = "粗定位(俯视)", STEP_FAR
                        elif phase == "outside":
                            res, top_ann, wr_ann = ask_mid(dec, top, wrist, pn[:2], letters)
                            stage, step_mm = "进环前(俯视+腕部)", STEP_MID
                        else:
                            res, wr_ann = ask_ring(dec, wrist, side, letters)
                            stage, step_mm = "环内(腕部+侧面)", STEP_RING
                        # 字母每次重新随机分配，所以投票用方向名，不用字母
                        a, b = (letters[x[-1]] if x.startswith("arrow_") else x
                                for x in (res["answers"]["a"]["choice"], res["answers"]["b"]["choice"]))
                        rec.update(stage=stage, answer=[a, b])
                        choice = vote(act_pool, a, b)
                        if choice is None:
                            action, note = "reask", note + f"{stage}：两次不一致 {a}/{b}"
                        else:
                            if a != b:
                                note += f"{MAX_REASK} 轮不一致，多数票；"
                            if choice in DIR_BASE:
                                action, delta = f"move_{choice}", np.array(DIR_BASE[choice]) * step_mm
                                note += f"{stage}：{choice} {step_mm:.0f}mm"
                            elif choice == "lower":
                                dz = max(pn[2] - STEP_DOWN, floor) - pn[2]
                                if dz > -1.0:
                                    action, note = "close", note + f"{stage}：要降，但已在法兰下限 {floor:.0f}mm → 在下限试合"
                                else:
                                    action, delta = "lower", (0, 0, dz)
                                    note += f"{stage}：下降 {-dz:.0f}mm"
                            elif choice == "close":
                                action, note = "close", note + f"{stage}：合手"
                            elif choice == "target_not_visible":
                                action, note = "stop", note + "俯视图里也看不到套筒"

                q_target = None
                if delta is not None:
                    q_target, reason = kin.solve(qn, pn + np.asarray(delta, float), R_g, floor)
                    if q_target is None:
                        note += f"；✗ 不能执行：{reason}"
                        action = f"{action}(被拒)"
                rec.update(action=action, note=note)
                print(f"step {step:3d}  t={rec['t']:5.1f}s  法兰 ({pn[0]:.0f},{pn[1]:.0f},{pn[2]:.0f})  -> {action}  | {note}")
                panel = CL.show([top_ann, wr_ann] + ([side] if side is not None else []),
                                f"step {step} -> {action}\n{note[:95]}", display=not args.mock)
                cv2.imwrite(str(out / f"{step:03d}_panel.jpg"), panel)
                log.append(rec)

                if action in ("stop", "done"):
                    result = "完成" if action == "done" else f"停止：{note}"
                    break
                if action.endswith("(被拒)"):
                    rejected += 1
                elif action != "reask":            # 重问没动，不算打破连续被拒
                    rejected = 0
                if rejected >= 3:
                    result = f"停止：连续 {rejected} 步都走不了（{note.split('；✗ 不能执行：')[-1]}）"
                    break
                if action == "close":
                    try_close(rec)
                    phase_pool.clear(), act_pool.clear()
                    if not closed and attempts >= MAX_ATTEMPTS:
                        result = f"停止：合手 {attempts} 次都空夹"
                        break
                    continue
                if q_target is not None:
                    rig.move(q_target, keys)
                    phase_pool.clear(), act_pool.clear()
                    if closed and action == "raise":
                        lifted += 1
            else:
                result = f"停止：到了 {MAX_STEPS} 步上限"

            print(f"\n结果：{result}（{step} 步，{time.time() - t_start:.0f}s，合手 {attempts} 次）")
            # 收尾：夹着就原路放下、张手；然后抬到起步高度、回起步点、倒着走起步路径回起始位
            if closed:
                for _ in range(lifted):
                    qn = rig.joints()
                    pn = kin.fk(qn)[:3, 3] * 1000
                    qt, reason = kin.solve(qn, pn - [0, 0, RAISE], R_g, floor)
                    if qt is None:
                        break
                    rig.move(qt, keys)
            rig.set_hand("open")
            print("回起始位 ...")
            qn = rig.joints()
            pn = kin.fk(qn)[:3, 3] * 1000
            ps = np.asarray(args.start, float)
            for target in (np.array([pn[0], pn[1], max(pn[2], ps[2])]), ps):
                while np.linalg.norm(target - pn) > 0.5:
                    nxt = pn + np.clip(target - pn, -20, 20)
                    qt, reason = kin.solve(qn, nxt, R_g, floor)
                    if qt is None:
                        raise RuntimeError(f"回程走不过去：{reason}")
                    rig.move(qt, keys)
                    qn, pn = qt, nxt
            gap = np.degrees(np.abs(rig.joints() - seq[-1])).max()
            if gap > CL.MAX_JUMP_DEG:          # 回到同一位姿却换了逆解支：不能直接 MoveAbsJ 甩过去
                raise RuntimeError(f"回到起步点时关节和去程差 {gap:.0f}°，换了逆解支，停在这里请手动回位")
            for qq in reversed(seq[:-1]):
                rig.move(qq, keys)
            rig.move(q_home, keys)
            print("已回到起始位。")
    except rc.StopRequested as e:
        result = f"急停：{e}"
        print(f"\n{result}。臂停在原地，手保持 {rig.hand_state}。")
    except Exception as e:  # noqa: BLE001
        result = f"出错：{type(e).__name__}: {e}"
        print(f"\n{result}。臂停在原地，手保持 {rig.hand_state}。")
    finally:
        summary = {"result": result, "steps": step, "seconds": round(time.time() - t_start, 1), "attempts": attempts,
                   "log": log}
        (out / "log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"记录 → {out}")
        cv2.destroyAllWindows()
        rig.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default=str(rc.DEFAULT_TRAJ), help="只用来取抓取朝向和示范抓取高度")
    ap.add_argument("--t-grasp", type=float, default=rc.DEFAULT_T_GRASP)
    ap.add_argument("--t-release", type=float, default=rc.DEFAULT_T_RELEASE)
    ap.add_argument("--step-mm", type=float, default=15.0)
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--start", type=float, nargs=3, default=(380.0, -300.0, 330.0))
    ap.add_argument("--out", default="~/atom_episodes/robotdecision_auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
