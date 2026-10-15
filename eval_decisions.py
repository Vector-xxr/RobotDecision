#!/usr/bin/env python3
"""用回放拍的数据离线测 OpenAI Decisions：两路画面 → 从预设动作里选一个。不碰任何硬件。

数据：replay_capture.py 的输出目录（meta.csv + top/ + wrist/）。每张图的「正确动作」由 FK 算：
  目标 G = 示范抓取位姿（P2_at_grasp 那一行的法兰位置）
  · P1_empty（白板上没套筒）            → target_not_visible
  · 起步段（示范 t<8.5s，手还在白板侧面那么低）→ 跟示范：示范下一步往哪走就是哪（先抬高）
  · 其余接近帧：水平差 >20mm → 朝 G 的水平方向（按俯视画面的上下左右）；
                水平已对准 → 高了 lower / 低了 raise / 差不到 15mm → close_hand
  · 合手之后（P4/P5）：比抓取高不到 100mm → raise，够了 → done
  斜着的情况（次轴分量 ≥ 主轴一半）两个方向都算对。

俯视画面方向 ↔ 基座：10-13 用 6 张回放图（手背白盒子像素 ↔ FK）拟合，
  画面上 = +x，下 = −x，左 = +y，右 = −y（镜像相似变换，残差中位 17px ≈ 12mm）。

    python eval_decisions.py --dry                  # 只算标签、打印一个请求，不调 API
    python eval_decisions.py                        # 全部帧 + 镜像/去说明两个子集
    python eval_decisions.py --variants base --limit 20

结果（每个请求原样落盘、可复算）→ RobotDecision/eval_runs/<时间>/
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPLAY = Path.home() / "atom_episodes/robotdecision_replay/20261013_103555"
KEY_FILE = Path.home() / ".config/openai/key"
MODEL = "gpt-6-luna"

# 俯视画面方向 → 基座方向（单位向量，xy）
IMG_DIR = {"move_up": (1, 0), "move_down": (-1, 0), "move_left": (0, 1), "move_right": (0, -1)}
XY_TOL, Z_TOL, LIFT_DONE = 20.0, 15.0, 100.0
EARLY_T = 8.5            # 示范这之前手在白板侧面偏低，要先抬

CONTEXT = (
    "You are choosing the next action for a robot arm with a two-finger robotic hand "
    "(a thumb and an index finger; the other fingers are not used).\n"
    "Image 1 is from a fixed TOP camera looking straight down at the work table: a white board "
    "lying on a green mat. The robot arm reaches in from the bottom-left of this image.\n"
    "Image 2 is from a WRIST camera fixed on the robot hand, looking down past the hand. "
    "The dark gray shapes at the left and right of the wrist image are the robot's own "
    "index finger and thumb; the object to grasp must end up between them.\n"
    "Task: pick up the blue sleeve (a short blue cylinder) from the white board.\n"
    "Each move action moves the hand a short step; directions refer to the TOP camera image."
)

ACTIONS = [
    ("move_up", "Move the hand a short step toward the TOP edge of the top-camera image."),
    ("move_down", "Move the hand a short step toward the BOTTOM edge of the top-camera image."),
    ("move_left", "Move the hand a short step toward the LEFT edge of the top-camera image."),
    ("move_right", "Move the hand a short step toward the RIGHT edge of the top-camera image."),
    ("lower", "Lower the hand a short step toward the table."),
    ("raise", "Raise the hand a short step away from the table."),
    ("close_hand", "Close the thumb and index finger to grasp. Only when the blue sleeve is "
                   "between the thumb and the index finger in the wrist image."),
    ("open_hand", "Open the thumb and index finger."),
    ("done", "The blue sleeve is held between the fingers and lifted clearly off the board."),
    ("target_not_visible", "The blue sleeve cannot be seen in either image."),
]
MIRROR_SWAP = {"move_left": "move_right", "move_right": "move_left"}

PREDICATES = [
    ("sleeve_visible", "Is a blue sleeve visible in either image?"),
    ("between_fingers", "In the wrist image, is the blue sleeve directly between the thumb and "
                        "the index finger, close enough that closing the fingers would grasp it?"),
    ("lifted", "Is the blue sleeve held by the hand and lifted off the board?"),
]


# ── 标签 ────────────────────────────────────────────────────────────────
def load_rows(replay):
    rows = list(csv.DictReader(open(replay / "meta.csv", encoding="utf-8")))
    for r in rows:
        r["p"] = np.array([float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])])
        r["src_t"] = float(r["src_t"])
    return rows


def horiz_actions(e_xy):
    """水平误差 → 主方向（+ 斜着时的次方向）。"""
    scores = {a: float(np.dot(e_xy, v)) for a, v in IMG_DIR.items()}
    order = sorted(scores, key=scores.get, reverse=True)
    main = order[0]
    ok = {main}
    if scores[order[1]] >= 0.5 * scores[main]:
        ok.add(order[1])
    return main, ok


def label(r, G, demo_next):
    ph = r["phase"]
    p = r["p"]
    if ph == "P1_empty":
        return "target_not_visible", {"target_not_visible"}, "无套筒"
    if ph in ("P4_closed", "P5_lift"):
        a = "done" if p[2] - G[2] >= LIFT_DONE else "raise"
        return a, {a}, "已合手"
    if ph == "P6_released":
        return None, set(), "放回后（不评）"
    if r["src_t"] < EARLY_T and demo_next is not None:      # 起步：跟示范
        d = demo_next - p
        if abs(d[2]) >= np.linalg.norm(d[:2]):
            a = "raise" if d[2] > 0 else "lower"
            return a, {a}, "起步跟示范"
        main, ok = horiz_actions(d[:2])
        return main, ok, "起步跟示范"
    e = G - p
    if np.linalg.norm(e[:2]) > XY_TOL:
        main, ok = horiz_actions(e[:2])
        return main, ok, "朝目标水平"
    if e[2] < -Z_TOL:
        return "lower", {"lower"}, "已对准，偏高"
    if e[2] > Z_TOL:
        return "raise", {"raise"}, "已对准，偏低"
    return "close_hand", {"close_hand"}, "到位"


def build_labels(rows):
    g_rows = [r for r in rows if r["phase"] == "P2_at_grasp"]
    # 网格数据没有 P2_at_grasp：目标位姿记在每行的 gx/gy/gz
    G = g_rows[0]["p"] if g_rows else np.array([float(rows[0][k]) for k in ("gx", "gy", "gz")])
    # 示范路径（P2 倒退那一串按示范时间排序）用来给起步段找「下一步」
    path = sorted([r for r in rows if r["phase"] in ("P2_back", "P2_at_grasp")], key=lambda r: r["src_t"])
    out = []
    for r in rows:
        nxt = None
        if r["phase"] in ("P2_back", "P2_at_grasp", "P1_empty"):
            later = [q for q in path if q["src_t"] > r["src_t"]]
            acc, prev = 0.0, r["p"]
            for q in later:
                acc += float(np.linalg.norm(q["p"] - prev))
                prev = q["p"]
                if acc >= 30:
                    nxt = q["p"]
                    break
        a, ok, why = label(r, G, nxt)
        r.update(gt=a, ok=ok, why=why)
        out.append(r)
    return out, G


# ── 请求 ────────────────────────────────────────────────────────────────
def img_url(path, mirror=False):
    if not mirror:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    import cv2
    img = cv2.flip(cv2.imread(str(path)), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def make_request(r, replay, variant):
    mirror = variant == "mirror"
    ctx = "" if variant == "nocontext" else CONTEXT + "\n"
    hand = "closed" if r["hand_state"] == "close" else "open"
    text = f"{ctx}Current hand state: {hand}."
    content = [{"type": "input_text", "text": text},
               {"type": "input_text", "text": "Image 1 (top camera):"},
               {"type": "input_image", "image_url": img_url(replay / r["top"], mirror), "detail": "auto"},
               {"type": "input_text", "text": "Image 2 (wrist camera):"},
               {"type": "input_image", "image_url": img_url(replay / r["wrist"], mirror), "detail": "auto"}]
    choices = [{"value": v, "description": d} for v, d in ACTIONS]
    instr = "Which single action should the robot take next to pick up the blue sleeve?"
    questions = [
        {"type": "choice", "name": "action_a", "instructions": instr, "choices": choices},
        {"type": "choice", "name": "action_b", "instructions": instr, "choices": choices[::-1]},
    ] + [{"type": "predicate", "name": n, "instructions": q} for n, q in PREDICATES]
    return {"model": MODEL, "input": [{"role": "user", "content": content}], "questions": questions}


# ── v2：三处改动（10-13 冒烟后） ──────────────────────────────────────────
# 1) 方向不叫 up/down：v1 里 move_up（俯视画面上方）和 raise（抬高）撞词，标 raise 的帧几乎全答成 move_up
# 2) 按手的开合只给相关选项（张开 8 个、合上 3 个），不是每次 10 选 1
# 3) 俯视图上从手的投影画 4 个带字母的箭头；字母和方向每帧随机配对，防止它记住「B=右」
IMG_OF = {"move_up": "north", "move_down": "south", "move_left": "west", "move_right": "east"}
FLIP = {"west": "east", "east": "west"}
DIR_DESC = {"north": "toward the TOP edge of image 1 (away from the robot base)",
            "south": "toward the BOTTOM edge of image 1 (toward the robot base)",
            "west": "toward the LEFT edge of image 1",
            "east": "toward the RIGHT edge of image 1"}
ARROW_VEC = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
ARROW_BGR = {"A": (0, 0, 255), "B": (0, 200, 255), "C": (255, 255, 0), "D": (255, 0, 255)}
# 法兰 (x, y) mm ↔ 手背白盒子像素（10-13 目测 6 张回放图），拟合俯视映射用
CAL_PTS = [((411, -10), (100, 245)), ((400, -106), (262, 242)), ((397, -168), (365, 250)),
           ((385, -241), (462, 255)), ((386, -297), (522, 258)), ((454, -159), (315, 212))]

CONTEXT_V2 = (
    "You are choosing the next action for a robot arm with a two-finger robotic hand "
    "(a thumb and an index finger).\n"
    "Image 1 is from a fixed TOP camera looking straight down at the work table: a white board "
    "lying on a green mat. The robot arm reaches in from the bottom-left of this image.\n"
    "Image 2 is from a WRIST camera fixed on the robot hand, looking down past the hand. "
    "The dark gray shapes at the left and right of the wrist image are the robot's own "
    "index finger and thumb; to grasp, the object must be between them.\n"
    "Task: pick up the blue sleeve (a short blue cylinder) from the white board.\n"
    "Horizontal moves keep the hand at the same height; 'raise' and 'lower' only change the height."
)


def fit_top_map():
    """法兰 xy → 俯视像素：镜像相似变换（相机朝下，画面纵轴和基座手性相反）。"""
    X = np.array([p[0] for p in CAL_PTS], float)
    U = np.array([p[1] for p in CAL_PTS], float)
    Y = X * [1, -1]
    mx, mu = Y.mean(0), U.mean(0)
    Uu, S, Vt = np.linalg.svd((U - mu).T @ (Y - mx))
    R = Uu @ Vt
    s = S.sum() / ((Y - mx) ** 2).sum()
    return lambda xy: s * R @ (np.asarray(xy, float) * [1, -1] - mx) + mu


TOP_MAP = fit_top_map()


def draw_arrows(img, origin, letters):
    """10-13 v3 冒烟：字母圈画在箭头尖端时压住了套筒（#114 因此答「看不到目标」0.91）。
    改成：箭头短（60px）、细；字母圈放在箭头中段的侧边、半透明 —— 箭头指向的那片地方不被挡。"""
    import cv2
    h, w = img.shape[:2]
    o = np.asarray(origin, float)
    for letter, d in letters.items():              # 先画箭头（不透明）
        v = np.array(ARROW_VEC[d], float)
        a = o + v * 15
        b = np.clip(o + v * 75, [10, 10], [w - 10, h - 10])
        col = ARROW_BGR[letter]
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (0, 0, 0), 5, tipLength=0.28)
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), col, 3, tipLength=0.28)
    labels = img.copy()                            # 再画字母圈，只有它半透明
    for letter, d in letters.items():
        v = np.array(ARROW_VEC[d], float)
        perp = np.array([v[1], -v[0]])            # 箭头左手边
        col = ARROW_BGR[letter]
        c = np.clip(o + v * 45 + perp * 17, [11, 11], [w - 11, h - 11]).astype(int)
        cv2.circle(labels, tuple(c), 10, (255, 255, 255), -1)
        cv2.circle(labels, tuple(c), 10, col, 2)
        cv2.putText(labels, letter, (int(c[0]) - 6, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    mask = np.any(labels != img, axis=2)
    img[mask] = (0.25 * img[mask] + 0.75 * labels[mask]).astype(np.uint8)
    return img


def make_request_v2(r, replay, arrows, mirror):
    """返回 (请求, 正确选项集合, 归类名)。"""
    import cv2

    closed = r["hand_state"] == "close"
    # 正确集合换到 v2 的说法（方向名 / 动作名）
    ok_dirs, ok_acts = set(), set()
    for a in r["ok"]:
        d = IMG_OF.get(a)
        if d:
            ok_dirs.add(FLIP.get(d, d) if mirror else d)
        else:
            ok_acts.add(a)
    g = IMG_OF.get(r["gt"], r["gt"])
    group = FLIP.get(g, g) if mirror else g

    top = cv2.imread(str(replay / r["top"]))
    wrist = cv2.imread(str(replay / r["wrist"]))
    if mirror:
        top, wrist = cv2.flip(top, 1), cv2.flip(wrist, 1)
    letters = {}
    if arrows and not closed:
        rng = np.random.default_rng(int(r["n"]) + (1000 if mirror else 0))
        letters = dict(zip("ABCD", rng.permutation(["north", "east", "south", "west"])))
        u, v = TOP_MAP(r["p"][:2])
        if mirror:
            u = top.shape[1] - 1 - u
        draw_arrows(top, (u, v), letters)

    def url(img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    if closed:
        choices = [("raise", "Raise the hand straight up a short step."),
                   ("done", "The blue sleeve is held between the fingers and lifted clearly off the board."),
                   ("open_hand", "Open the thumb and index finger.")]
    else:
        if letters:
            choices = [(f"arrow_{k}", f"Move the hand a short step along arrow {k} drawn in image 1.")
                       for k in "ABCD"]
            ok_vals = {f"arrow_{k}" for k, d in letters.items() if d in ok_dirs}
        else:
            choices = [(f"move_{d}", f"Move the hand a short step horizontally {DIR_DESC[d]}.")
                       for d in ("north", "east", "south", "west")]
            ok_vals = {f"move_{d}" for d in ok_dirs}
        choices += [("lower", "Lower the hand straight down a short step toward the table."),
                    ("raise", "Raise the hand straight up a short step."),
                    ("close_hand", "Close the thumb and index finger to grasp. Only when the blue sleeve "
                                   "is between the thumb and the index finger in the wrist image."),
                    ("target_not_visible", "The blue sleeve cannot be seen in either image.")]
    if closed:
        ok_vals = set()
    ok_vals |= ok_acts

    text = CONTEXT_V2
    if letters:
        text += ("\nIn image 1, four arrows labeled A, B, C and D start at the robot hand; each shows "
                 "one horizontal direction the hand can move.")
    text += f"\nCurrent hand state: {'closed' if closed else 'open'}."
    content = [{"type": "input_text", "text": text},
               {"type": "input_text", "text": "Image 1 (top camera):"},
               {"type": "input_image", "image_url": url(top), "detail": "auto"},
               {"type": "input_text", "text": "Image 2 (wrist camera):"},
               {"type": "input_image", "image_url": url(wrist), "detail": "auto"}]
    ch = [{"value": v, "description": d} for v, d in choices]
    instr = "Which single action should the robot take next to pick up the blue sleeve?"
    questions = [
        {"type": "choice", "name": "action_a", "instructions": instr, "choices": ch},
        {"type": "choice", "name": "action_b", "instructions": instr, "choices": ch[::-1]},
    ] + [{"type": "predicate", "name": n, "instructions": q} for n, q in PREDICATES]
    req = {"model": MODEL, "input": [{"role": "user", "content": content}], "questions": questions}
    return req, ok_vals, group, top


# ── v3：腕部目标圈（辅助 Decisions，不是规则） + 「完成」交给规则 + 标签只看能不能抓到 ──────
# 圈 = 手刚好在抓取位姿时套筒在腕部画面里的位置和大小（10-13 回放 #92 量的：中心 (304,228) 半径 110px）。
# 腕部相机和手是刚性的，所以这是一次性标定，运行时不检测套筒。高 18~29mm 时套筒半径只有 89~96px。
WRIST_CIRCLE = ((304, 228), 110)
CLOSE_DZ, MAYBE_DZ = 15.0, 25.0     # 高出 ≤15mm 该合；15~25mm 合或降都算对（能不能抓到说不准）；再高该降

CONTEXT_V3 = (
    "You are choosing the next action for a robot arm with a two-finger robotic hand "
    "(a thumb and an index finger).\n"
    "Image 1 is from a fixed TOP camera looking straight down at the work table: a white board "
    "lying on a green mat. The robot arm reaches in from the bottom-left of this image. Four arrows "
    "labeled A, B, C and D start at the robot hand; each shows one horizontal direction the hand can move.\n"
    "Image 2 is from a WRIST camera fixed on the robot hand, looking down past the hand. The dark gray "
    "shape at the right edge is the thumb; when the hand is open the index finger is mostly outside the "
    "image on the left. The GREEN CIRCLE in image 2 shows where the object must appear, and how large it "
    "must look, for the fingers to grasp it: if the blue sleeve fills the green circle, close the hand; "
    "if the blue sleeve is inside the circle but clearly smaller than it, the hand is too high, lower it.\n"
    "Task: pick up the blue sleeve (a short blue cylinder) from the white board.\n"
    "Horizontal moves keep the hand at the same height; 'raise' and 'lower' only change the height."
)
PREDICATES_V3_OPEN = [
    ("sleeve_visible", "Is a blue sleeve visible in either image?"),
    ("fills_circle", "In image 2, is the blue sleeve inside the green circle and about as large as the circle?"),
]
PREDICATES_V3_CLOSED = [
    ("sleeve_visible", "Is a blue sleeve visible in either image?"),
    ("holding", "In image 2, is the blue sleeve held between the thumb and the index finger?"),
]


def label_v3(r, G):
    """只看「这一步能不能让手更接近抓到」，不要求照着示范的路走（用户 10-13：遥操只是一条示例）。"""
    p = r["p"]
    if r["phase"] == "P1_empty":
        return {"target_not_visible"}, "target_not_visible"
    e = G - p
    dz = p[2] - G[2]
    if np.linalg.norm(e[:2]) > XY_TOL:
        main, ok = horiz_actions(e[:2])
        dirs = {IMG_OF[a] for a in ok}
        if r["src_t"] < EARLY_T:          # 起步时手还在白板侧面偏低：先抬也算对
            return dirs | {"raise"}, IMG_OF[main]
        return dirs, IMG_OF[main]
    # 水平已在 2cm 内但还偏 8mm 以上（腕部画面里套筒会偏出圈）：先朝它挪也算对
    near = set()
    if np.linalg.norm(e[:2]) > 8.0:
        near = {IMG_OF[horiz_actions(e[:2])[0]]}
    if dz <= CLOSE_DZ:
        return {"close_hand"} | near, "close_hand"
    if dz <= MAYBE_DZ:
        return {"close_hand", "lower"} | near, "lower_or_close"
    return {"lower"} | near, "lower"


SIDE_TEXT = ("Image 3 is from a SIDE camera on the near edge of the table, looking across the white board "
             "toward the robot. It shows the hand's fingertips and the blue sleeve from the side, including "
             "how high the fingertips are above the board and which way the sleeve is from the fingertips.")


# ── v5：腕部画面也画同字母箭头 ───────────────────────────────────────────────
# 10-13 网格采集（套筒上方 +50/+80mm、5×5、±20mm）标的：套筒像素 = c0 + J·(手水平偏移)，残差 1.4~2.9px。
# 手相对场景在腕部画面里的移动方向 = −J·d。腕部画面相对俯视转了约 45° 且镜像，模型自己换算不过来。
WRIST_J = np.array([[1.89, -1.85], [-1.95, -1.84]])        # G50 层，px/mm，列 = 基座 x、y
WRIST_VEC = {}
for _d, _v in (("north", (1, 0)), ("south", (-1, 0)), ("west", (0, 1)), ("east", (0, -1))):
    _h = -(WRIST_J @ np.array(_v, float))
    WRIST_VEC[_d] = _h / np.linalg.norm(_h)                 # 图像 (u, v)，v 向下

V5_WRIST_TEXT = (
    "In image 2 the same four lettered arrows start at the edge of the green circle; arrow X in image 2 is the "
    "same move as arrow X in image 1 (they point differently because the wrist camera is turned). When the blue "
    "sleeve is visible in image 2 but not inside the green circle, choose the arrow in image 2 that points from "
    "the circle toward the sleeve.")


def draw_wrist_arrows(img, centre, letters, mirror):
    import cv2
    h, w = img.shape[:2]
    o = np.asarray(centre, float)
    rad = WRIST_CIRCLE[1]
    for letter, d in letters.items():
        v = WRIST_VEC[d] * ([-1, 1] if mirror else [1, 1])
        a = np.clip(o + v * (rad + 8), [10, 10], [w - 10, h - 10])
        b = np.clip(o + v * (rad + 70), [10, 10], [w - 10, h - 10])
        col = ARROW_BGR[letter]
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (0, 0, 0), 5, tipLength=0.3)
        cv2.arrowedLine(img, tuple(int(x) for x in a), tuple(int(x) for x in b), col, 3, tipLength=0.3)
        perp = np.array([v[1], -v[0]])
        c = np.clip(o + v * (rad + 40) + perp * 17, [11, 11], [w - 11, h - 11]).astype(int)
        cv2.circle(img, tuple(c), 10, (255, 255, 255), -1)
        cv2.circle(img, tuple(c), 10, col, 2)
        cv2.putText(img, letter, (int(c[0]) - 6, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


def label_grid(r, G):
    """网格帧（套筒上方 +50/+80mm）：水平偏 ≥10mm → 朝套筒水平移（斜的两个方向都算）；否则 → 下降。"""
    e = G - r["p"]
    if np.linalg.norm(e[:2]) >= 10.0:
        main, ok = horiz_actions(e[:2])
        return {IMG_OF[a] for a in ok}, IMG_OF[main]
    return {"lower"}, "lower"


def make_request_v3(r, replay, mirror, G, side=False, wrist_arrows=False):
    """返回 (请求, 正确选项集合, 归类名, 各 predicate 的真值)。手合上时不问动作，只问抓没抓住。"""
    import cv2

    closed = r["hand_state"] == "close"
    top = cv2.imread(str(replay / r["top"]))
    wrist = cv2.imread(str(replay / r["wrist"]))
    if mirror:
        top, wrist = cv2.flip(top, 1), cv2.flip(wrist, 1)
    (cu, cv_), rad = WRIST_CIRCLE
    if mirror:
        cu = wrist.shape[1] - 1 - cu
    cv2.circle(wrist, (cu, cv_), rad, (0, 0, 0), 6)
    cv2.circle(wrist, (cu, cv_), rad, (0, 255, 0), 3)

    truth = {"sleeve_visible": r["phase"] != "P1_empty"}
    if closed:
        ok_vals, group, ch = set(), "holding_check", None
        truth["holding"] = r["phase"] in ("P4_closed", "P5_lift")
    else:
        ok, group = label_grid(r, G) if r["phase"].startswith("G") else label_v3(r, G)
        if mirror:
            ok = {FLIP.get(a, a) for a in ok}
            group = FLIP.get(group, group)
        rng = np.random.default_rng(int(r["n"]) + (1000 if mirror else 0))
        letters = dict(zip("ABCD", rng.permutation(["north", "east", "south", "west"])))
        u, v = TOP_MAP(r["p"][:2])
        if mirror:
            u = top.shape[1] - 1 - u
        draw_arrows(top, (u, v), letters)
        if wrist_arrows:
            draw_wrist_arrows(wrist, (cu, cv_), letters, mirror)
        ok_vals = {f"arrow_{k}" for k, d in letters.items() if d in ok} | (ok - set(ARROW_VEC))
        choices = [(f"arrow_{k}", f"Move the hand a short step along arrow {k} drawn in image 1.")
                   for k in "ABCD"]
        choices += [("lower", "Lower the hand straight down a short step toward the table."),
                    ("raise", "Raise the hand straight up a short step."),
                    ("close_hand", "Close the thumb and index finger to grasp the blue sleeve."),
                    ("target_not_visible", "The blue sleeve cannot be seen in either image.")]
        ch = [{"value": a, "description": d} for a, d in choices]
        # 15~25mm 那段套筒半径只有 ~90px（圈 110）：动作合/降都算对，但「填满圈」这题不计分
        truth["fills_circle"] = None if group == "lower_or_close" else ("close_hand" in ok)

    def url(img):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    text = (CONTEXT_V3 + ("\n" + V5_WRIST_TEXT if wrist_arrows and not closed else "")
            + ("\n" + SIDE_TEXT if side else "") + f"\nCurrent hand state: {'closed' if closed else 'open'}.")
    content = [{"type": "input_text", "text": text},
               {"type": "input_text", "text": "Image 1 (top camera):"},
               {"type": "input_image", "image_url": url(top), "detail": "auto"},
               {"type": "input_text", "text": "Image 2 (wrist camera):"},
               {"type": "input_image", "image_url": url(wrist), "detail": "auto"}]
    if side:
        sim = cv2.imread(str(replay / r["side"]))
        if mirror:
            sim = cv2.flip(sim, 1)
        content += [{"type": "input_text", "text": "Image 3 (side camera):"},
                    {"type": "input_image", "image_url": url(sim), "detail": "auto"}]
    preds = PREDICATES_V3_CLOSED if closed else PREDICATES_V3_OPEN
    questions = [{"type": "predicate", "name": n, "instructions": q} for n, q in preds]
    if ch:
        instr = "Which single action should the robot take next to pick up the blue sleeve?"
        questions = [{"type": "choice", "name": "action_a", "instructions": instr, "choices": ch},
                     {"type": "choice", "name": "action_b", "instructions": instr, "choices": ch[::-1]}] + questions
    req = {"model": MODEL, "input": [{"role": "user", "content": content}], "questions": questions}
    return req, ok_vals, group, truth, top, wrist


def summarize_v3(results, variant):
    rows = [r for r in results if r["variant"] == variant]
    if not rows:
        return []
    act = [r for r in rows if r["gt"] != "holding_check"]
    lines = [f"\n## {variant}（{len(rows)} 帧，其中选动作的 {len(act)} 帧）"]
    if act:
        acc_a = np.mean([r["action_a"] in r["ok"] for r in act])
        acc_b = np.mean([r["action_b"] in r["ok"] for r in act])
        agree = np.mean([r["action_a"] == r["action_b"] for r in act])
        both = np.mean([r["action_a"] == r["action_b"] and r["action_a"] in r["ok"] for r in act])
        lines.append(f"- 选择题本身：正序 {acc_a:.0%}、倒序 {acc_b:.0%}；两遍一致 {agree:.0%}；一致且对 {both:.0%}")

        # 实际要用的策略：fills_circle ≥ 0.5 → 合手；否则用选择题（选择题说合手但圈没填满 → 换倒序那个，再不行就不动）
        def policy(r):
            f = r.get("pred_fills_circle")
            if f is not None and f >= 0.5:
                return "close_hand"
            for a in (r["action_a"], r["action_b"]):
                if a and a != "close_hand":
                    return a
            return None
        pol = [(policy(r), r) for r in act]
        dec = [(a, r) for a, r in pol if a]
        lines.append(f"- **策略（合手由「填满圈」触发）**：出动作 {len(dec)}/{len(act)}，"
                     f"对 {np.mean([a in r['ok'] for a, r in dec]):.0%}")
        byp = defaultdict(list)
        for a, r in dec:
            byp[r["gt"]].append((a, r))
        for g, rs in sorted(byp.items(), key=lambda kv: -len(kv[1])):
            c = Counter(a for a, _ in rs)
            lines.append(f"  - {g}（{len(rs)}）对 {np.mean([a in r['ok'] for a, r in rs]):.0%}；出成 " +
                         "、".join(f"{k}×{v}" for k, v in c.most_common(4)))
        by = defaultdict(list)
        for r in act:
            by[r["gt"]].append(r)
        lines.append("- 按正确动作分（方向按俯视画面：east=右、west=左）：")
        for g, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            c = Counter(r["action_a"] for r in rs)
            hit = np.mean([r["action_a"] in r["ok"] for r in rs])
            lines.append(f"  - {g}（{len(rs)}）对 {hit:.0%}；答成 " +
                         "、".join(f"{k}×{v}" for k, v in c.most_common(4)))
    for name in ("sleeve_visible", "fills_circle", "holding"):
        ps = [(r.get(f"pred_{name}"), r.get(f"truth_{name}")) for r in rows
              if r.get(f"pred_{name}") is not None and r.get(f"truth_{name}") is not None]
        if ps:
            pos = [p for p, t in ps if t]
            neg = [p for p, t in ps if not t]
            acc = np.mean([(p >= 0.5) == t for p, t in ps])
            lines.append(f"- predicate {name}：按 0.5 判对 {acc:.0%}；真为是 {len(pos)} 帧概率中位 "
                         f"{np.median(pos) if pos else float('nan'):.2f}，真为否 {len(neg)} 帧中位 "
                         f"{np.median(neg) if neg else float('nan'):.2f}")
    ms = [r["ms"] for r in rows]
    lines.append(f"- 延迟 p50 {np.percentile(ms, 50):.0f} ms、p95 {np.percentile(ms, 95):.0f} ms；"
                 f"输入 token 中位 {np.median([r['input_tokens'] for r in rows]):.0f}")
    return lines


class DailyLimit(Exception):
    pass


class Throttle:
    """按每分钟请求数匀速放行。免费档 gpt-6-luna 是 10 RPM / 60k TPM / 每天 50 次（10-13 实测报错原文）。"""

    def __init__(self, rpm):
        import threading
        self.gap = 60.0 / max(rpm, 0.1)
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def call(client, req, throttle, retries=6):
    import re
    from openai import RateLimitError

    for attempt in range(retries):
        throttle.wait()
        t = time.perf_counter()
        try:
            d = client.decisions.create(**req)
            break
        except RateLimitError as e:
            msg = str(e)
            if "per day" in msg:
                raise DailyLimit(msg) from e
            m = re.search(r"try again in ([\d.]+)(ms|s)", msg)
            wait = float(m.group(1)) / (1000 if m and m.group(2) == "ms" else 1) if m else 6.0
            time.sleep(wait + 0.5 + attempt)
    else:
        raise RuntimeError(f"限流重试 {retries} 次仍失败")
    ms = (time.perf_counter() - t) * 1000
    ans = {}
    for a in d.answers:
        x = a.model_dump()
        ans[a.name] = x
    return {"ms": ms, "input_tokens": d.usage.input_tokens, "answers": ans}


# ── 评估 ────────────────────────────────────────────────────────────────
def summarize(results, variant):
    rows = [r for r in results if r["variant"] == variant and r.get("gt")]
    if not rows:
        return []
    lines = [f"\n## {variant}（{len(rows)} 帧）"]
    swap = MIRROR_SWAP if variant == "mirror" else {}

    def okset(r):
        return {swap.get(a, a) for a in r["ok"]}

    acc_a = np.mean([r["action_a"] in okset(r) for r in rows])
    acc_b = np.mean([r["action_b"] in okset(r) for r in rows])
    agree = np.mean([r["action_a"] == r["action_b"] for r in rows])
    both = np.mean([r["action_a"] == r["action_b"] and r["action_a"] in okset(r) for r in rows])
    lines.append(f"- 动作正确率：正序 {acc_a:.0%}、倒序 {acc_b:.0%}；两遍一致 {agree:.0%}；"
                 f"一致且对 {both:.0%}")
    by = defaultdict(list)
    for r in rows:
        by[swap.get(r["gt"], r["gt"])].append(r)
    lines.append("- 按正确动作分：")
    for g, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        c = Counter(r["action_a"] for r in rs)
        hit = np.mean([r["action_a"] in okset(r) for r in rs])
        lines.append(f"  - {g}（{len(rs)}）对 {hit:.0%}；答成 " +
                     "、".join(f"{k}×{v}" for k, v in c.most_common(4)))
    conf = [r["conf_a"] for r in rows if r["conf_a"] is not None]
    wrong_conf = [r["conf_a"] for r in rows if r["action_a"] not in okset(r) and r["conf_a"] is not None]
    if conf:
        lines.append(f"- confidence 中位 {np.median(conf):.2f}；答错时中位 "
                     f"{np.median(wrong_conf) if wrong_conf else float('nan'):.2f}（{len(wrong_conf)} 次答错）")
    for name, truth in (("sleeve_visible", lambda r: r["phase"] != "P1_empty"),
                        ("between_fingers", lambda r: r["gt"] == "close_hand" or r["phase"] == "P4_closed"),
                        ("lifted", lambda r: r["gt"] == "done")):
        ps = [(r[name], truth(r)) for r in rows if r.get(name) is not None]
        if ps:
            pos = [p for p, t in ps if t]
            neg = [p for p, t in ps if not t]
            acc = np.mean([(p >= 0.5) == t for p, t in ps])
            lines.append(f"- predicate {name}：按 0.5 判对 {acc:.0%}；真为是 {len(pos)} 帧概率中位 "
                         f"{np.median(pos) if pos else float('nan'):.2f}，真为否 {len(neg)} 帧中位 "
                         f"{np.median(neg) if neg else float('nan'):.2f}")
    ms = [r["ms"] for r in rows]
    toks = [r["input_tokens"] for r in rows]
    lines.append(f"- 延迟 p50 {np.percentile(ms, 50):.0f} ms、p95 {np.percentile(ms, 95):.0f} ms、"
                 f"最大 {max(ms):.0f} ms；输入 token 中位 {np.median(toks):.0f}")
    return lines


def stratify(jobs, budget, rng):
    """按（变体, 正确动作）分组，轮流各抽一个，直到凑够 budget —— 小预算下每类都有样本。"""
    groups = defaultdict(list)
    for v, r in jobs:
        groups[(v, r["gt"])].append((v, r))
    for g in groups.values():
        rng.shuffle(g)
    picked = []
    while len(picked) < budget and any(groups.values()):
        for k in sorted(groups, key=lambda k: (k[0] != "base", k)):
            if groups[k] and len(picked) < budget:
                picked.append(groups[k].pop())
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--variants", default="base,mirror,nocontext")
    ap.add_argument("--subset", type=int, default=40, help="mirror / nocontext 只抽这么多张接近帧")
    ap.add_argument("--limit", type=int, default=0, help="base 也只跑前 N 帧（调试用）")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rpm", type=float, default=9, help="每分钟最多发几个（免费档上限 10）")
    ap.add_argument("--budget", type=int, default=0,
                    help="只跑这么多个请求，按类别分层抽样（免费档每天 50 次时用）")
    ap.add_argument("--resume", default="", help="接着某次 eval_runs/<时间> 跑，已有回答的不再发")
    ap.add_argument("--frames", default="", help="只用这些帧号（逗号分隔），v2 对比时和上一轮用同一批")
    ap.add_argument("--near", type=float, default=0, help="只用离目标水平距离小于这么多 mm 的有套筒接近帧")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    replay = Path(args.replay).expanduser()
    rows, G = build_labels(load_rows(replay))
    rated = [r for r in rows if r["gt"]]
    print(f"数据 {replay}：{len(rows)} 帧，参与评估 {len(rated)} 帧；目标 G = {np.round(G).tolist()} mm")
    print("正确动作分布：", dict(Counter(r["gt"] for r in rated)))
    print("来源：", dict(Counter(r["why"] for r in rows)))

    variants = args.variants.split(",")
    approach = [r for r in rated if r["phase"] in ("P2_back", "P2_at_grasp") or r["phase"].startswith("G")]
    rng = np.random.default_rng(0)
    sub = [approach[i] for i in sorted(rng.choice(len(approach), min(args.subset, len(approach)), replace=False))]
    jobs = []
    if args.near:
        args.frames = ",".join(str(r["n"]) for r in rated if r["phase"] in ("P2_back", "P2_at_grasp")
                                 and np.linalg.norm((G - r["p"])[:2]) < args.near)
    if args.frames:
        want = {int(x) for x in args.frames.split(",")}
        chosen = [r for r in rated if int(r["n"]) in want]
        for v in variants:
            mir = "mirror" in v
            jobs += [(v, r) for r in chosen if not mir or r in approach]
    else:
        for v in variants:
            if v.startswith(("v3", "v4", "v5")):          # 全量：原图全部帧，镜像全部接近帧
                frames = approach if "mirror" in v else rated
            else:
                frames = (rated[:args.limit] if args.limit else rated) if v == "base" else sub
            jobs += [(v, r) for r in frames]
    if args.budget:
        jobs = stratify(jobs, args.budget, rng)
    print(f"请求数 {len(jobs)}（{dict(Counter(v for v, _ in jobs))}）")
    if args.dry and any(v.startswith("v3") for v in variants):
        import cv2
        out = HERE / "eval_runs" / "_preview"
        out.mkdir(parents=True, exist_ok=True)
        for v, r in jobs:
            req, okv, grp, truth, top, wrist = make_request_v3(r, replay, "mirror" in v, G)
            if int(r["n"]) in (92, 97, 101, 114, 193) and "mirror" not in v:
                cv2.imwrite(str(out / f"v3_{int(r['n']):04d}.jpg"),
                            np.hstack([cv2.resize(top, (480, 360)), cv2.resize(wrist, (480, 360))]))
            print(f"  {v} #{r['n']} {r['phase']} dz={r['p'][2]-G[2]:+.0f} 归类 {grp} 正确 {sorted(okv)} "
                  f"真值 {truth}")
        print(f"预览图 → {out}")
        return 0
    if args.dry and any(v.startswith("v2") for v in variants):
        import cv2
        out = HERE / "eval_runs" / "_preview"
        out.mkdir(parents=True, exist_ok=True)
        for v, r in jobs[:: max(1, len(jobs) // 6)][:6]:
            req, okv, grp, top = make_request_v2(r, replay, "arrows" in v, "mirror" in v)
            cv2.imwrite(str(out / f"{v}_{int(r['n']):04d}.jpg"), top)
            print(f"  {v} #{r['n']} {r['phase']} 归类 {grp} 正确 {sorted(okv)} 选项 "
                  f"{[c['value'] for c in req['questions'][0]['choices']]}")
        print(f"预览图 → {out}")
        return 0
    if args.dry:
        req = make_request(rated[0], replay, "base")
        for c in req["input"][0]["content"]:
            if c["type"] == "input_image":
                c["image_url"] = c["image_url"][:40] + "..."
        print(json.dumps(req, ensure_ascii=False, indent=1)[:3000])
        return 0

    from openai import OpenAI
    client = OpenAI(api_key=KEY_FILE.read_text().strip(), max_retries=0)   # 限流自己处理
    run_dir = Path(args.resume) if args.resume else HERE / "eval_runs" / time.strftime("%Y%m%d_%H%M%S")
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    throttle = Throttle(args.rpm)
    stop = {"daily": None}
    results = []

    def work(v, r):
        cache = run_dir / "responses" / f"{v}_{int(r['n']):04d}.json"
        if v.startswith(("v3", "v4", "v5")):
            req, ok_vals, group, truth, _, _ = make_request_v3(r, replay, "mirror" in v, G, side=v.startswith("v4"),
                                                                wrist_arrows=v.startswith("v5"))
            if cache.exists():
                res = json.loads(cache.read_text(encoding="utf-8"))
            else:
                res = call(client, req, throttle)
                cache.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            a = res["answers"]

            def g3(name, key):
                x = a.get(name)
                return None if not x or x.get("type") == "refusal" else x.get(key)
            out = {"variant": v, "n": int(r["n"]), "phase": r["phase"], "src_t": r["src_t"],
                   "gt": group, "ok": ok_vals, "why": r["why"],
                   "action_a": g3("action_a", "choice"), "action_b": g3("action_b", "choice"),
                   "conf_a": g3("action_a", "confidence"),
                   "ms": res["ms"], "input_tokens": res["input_tokens"],
                   "x": r["p"][0], "y": r["p"][1], "z": r["p"][2]}
            for name in ("sleeve_visible", "fills_circle", "holding"):
                out[f"pred_{name}"] = g3(name, "probability")
                out[f"truth_{name}"] = truth.get(name)
            return out
        if v.startswith("v2"):
            req, ok_vals, group, _ = make_request_v2(r, replay, "arrows" in v, "mirror" in v)
        else:
            req, ok_vals, group = None, r["ok"], r["gt"]
        if cache.exists():
            res = json.loads(cache.read_text(encoding="utf-8"))
        else:
            if stop["daily"]:
                raise DailyLimit(stop["daily"])
            res = call(client, req or make_request(r, replay, v), throttle)
            cache.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        a = res["answers"]

        def get(name, key):
            x = a.get(name, {})
            return x.get(key) if x.get("type") != "refusal" else None
        return {"variant": v, "n": int(r["n"]), "phase": r["phase"], "src_t": r["src_t"],
                "gt": group, "ok": ok_vals, "why": r["why"],
                "action_a": get("action_a", "choice"), "action_b": get("action_b", "choice"),
                "conf_a": get("action_a", "confidence"), "conf_b": get("action_b", "confidence"),
                **{n: get(n, "probability") for n, _ in PREDICATES},
                "ms": res["ms"], "input_tokens": res["input_tokens"],
                "x": r["p"][0], "y": r["p"][1], "z": r["p"][2]}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, v, r) for v, r in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except DailyLimit as e:
                if not stop["daily"]:
                    stop["daily"] = str(e)
                    print(f"\n  ✗ 到了每天的请求上限，后面的不再发。进度已存，之后用 --resume {run_dir} 接着跑")
            except Exception as e:  # noqa: BLE001
                print(f"\n  ✗ {type(e).__name__}: {e}")
            print(f"\r  {i}/{len(jobs)}  {time.time()-t0:.0f}s", end="", flush=True)
    print()
    if not results:
        print("一个回答都没有。")
        return 1

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        keys = [k for k in results[0] if k != "ok"] + ["ok"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["variant"], r["n"])):
            w.writerow({**r, "ok": "|".join(sorted(r["ok"]))})
    lines = [f"# Decisions 离线评测 {run_dir.name}", f"数据 {replay}，模型 {MODEL}，目标 G={np.round(G).tolist()}"]
    for v in variants:
        lines += summarize_v3(results, v) if v.startswith(("v3", "v4", "v5")) else summarize(results, v)
    total_tok = sum(r["input_tokens"] for r in results)
    lines.append(f"\n共 {len(results)} 个请求，输入 {total_tok} token ≈ ${total_tok / 1e6 * 0.10:.3f}")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
