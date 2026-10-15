#!/usr/bin/env python3
"""两个只用腕部画面的离线小实验（10-13，网格采集之后）。

A1 近处方向：腕部图上画绿圈 + 4 根同字母箭头，只问「哪根箭头从圈指向套筒中心 / 已对正 / 看不到」。
   v5 把这件事混在「下一步做什么」里问，网格上只有 35% —— 这里看问题拆简单后能到多少。
   只有一张图，所以左右翻转（画面和箭头一起翻）仍是自洽场景，可以当「在不在看图」的检验。
A2 防过早合手：手在套筒正上方 5~8cm 时「填满圈」误报 24~42%。比两种问法：
   single = 现在的单圈问法；rings = 外圈 + 虚线内圈（「套筒只有内圈这么大 = 还高约 3cm」）。

    python near_tests.py --dry           # 只出标签和预览图
    python near_tests.py                 # 跑 A1 + A2
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_decisions as E  # noqa: E402

GRID = sorted(d for d in (Path.home() / "atom_episodes/robotdecision_grid").glob("2026*") if not d.name.endswith("mock"))[-1]
REPLAYS = [Path.home() / "atom_episodes/robotdecision_replay/20261013_122101",
           Path.home() / "atom_episodes/robotdecision_replay/20261013_103555"]
INNER_R = 88          # 高约 +25~30mm 时套筒在腕部画面里的半径（10-13 实测 89~96px）

A1_TEXT = ("This image is from a camera fixed on a robot hand, looking down at a table. The GREEN CIRCLE marks "
           "where an object must be for the fingers to grasp it. Four lettered arrows start at the circle; each "
           "shows one direction the hand can move, and moving along an arrow brings the circle toward whatever "
           "that arrow points at.")
A1_Q = ("Which arrow points from the green circle toward the center of the blue sleeve? Choose 'centered' if the "
        "sleeve's center is already near the circle's center, or 'not_visible' if no blue sleeve is visible.")
A2_SINGLE_TEXT = ("This image is from a camera fixed on a robot hand, looking down at a table. The GREEN CIRCLE shows "
                  "where, and how large, an object must appear for the fingers to grasp it.")
A2_SINGLE_Q = "Is the blue sleeve inside the green circle and about as large as the circle?"
A2_RINGS_TEXT = ("This image is from a camera fixed on a robot hand, looking down at a table. The solid GREEN CIRCLE "
                 "shows how large the blue sleeve looks when the fingers are at the right height to grasp it. The "
                 "smaller DASHED circle shows how large it looks when the hand is still about 3 cm too high.")
A2_RINGS_Q = ("Does the blue sleeve look as large as the solid green circle, with its outer edge close to the solid "
              "circle all the way around (clearly larger than the dashed circle)?")


def blob_area(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return int(((hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)).sum())


def frames():
    """[(来源, 行, G)]：网格全部 + 两次回放里有套筒、离目标 <60mm 的接近帧。"""
    out = []
    rows, G = E.build_labels(E.load_rows(GRID))
    out += [(GRID, r, G) for r in rows]
    for rp in REPLAYS:
        rows, G = E.build_labels(E.load_rows(rp))
        out += [(rp, r, G) for r in rows if r["phase"] in ("P2_back", "P2_at_grasp")
                and np.linalg.norm((G - r["p"])[:2]) < 60]
    return out


def a1_label(src, r, G):
    e = G - r["p"]
    off = np.linalg.norm(e[:2])
    if r["phase"].startswith("P") and blob_area(cv2.imread(str(src / r["wrist"]))) < 1500:
        return {"not_visible"}, "not_visible"
    if off < 8:
        return {"centered"}, "centered"
    main, ok = E.horiz_actions(e[:2])
    dirs = {E.IMG_OF[a] for a in ok}
    if off < 12:
        return dirs | {"centered"}, E.IMG_OF[main]
    return dirs, E.IMG_OF[main]


def a2_truth(r, G):
    e = G - r["p"]
    dz = r["p"][2] - G[2]
    if dz <= 15 and np.linalg.norm(e[:2]) <= 10:
        return True
    if dz >= 25:
        return False
    return None                      # 15~25mm 说不清，不计分


def img_url(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def wrist_img(src, r, mirror=False, arrows=None, rings=False):
    img = cv2.imread(str(src / r["wrist"]))
    if mirror:
        img = cv2.flip(img, 1)
    (cu, cv_), rad = E.WRIST_CIRCLE
    if mirror:
        cu = img.shape[1] - 1 - cu
    cv2.circle(img, (cu, cv_), rad, (0, 0, 0), 6)
    cv2.circle(img, (cu, cv_), rad, (0, 255, 0), 3)
    if rings:
        for k in range(0, 360, 20):                       # 虚线内圈
            cv2.ellipse(img, (cu, cv_), (INNER_R, INNER_R), 0, k, k + 10, (0, 255, 0), 2)
    if arrows:
        E.draw_wrist_arrows(img, (cu, cv_), arrows, mirror)
    return img


def a1_request(src, r, G, mirror):
    rng = np.random.default_rng(int(r["n"]) + (1000 if mirror else 0) + (hash(src.name) % 997))
    letters = dict(zip("ABCD", rng.permutation(["north", "east", "south", "west"])))
    ok, group = a1_label(src, r, G)
    ok_vals = {f"arrow_{k}" for k, d in letters.items() if d in ok} | (ok & {"centered", "not_visible"})
    img = wrist_img(src, r, mirror, arrows=letters)
    ch = [{"value": f"arrow_{k}", "description": f"Arrow {k}."} for k in "ABCD"]
    ch += [{"value": "centered", "description": "The sleeve is already centered on the green circle."},
           {"value": "not_visible", "description": "No blue sleeve is visible in the image."}]
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": A1_TEXT},
                                                  {"type": "input_image", "image_url": img_url(img), "detail": "auto"}]}],
           "questions": [{"type": "choice", "name": "a", "instructions": A1_Q, "choices": ch},
                         {"type": "choice", "name": "b", "instructions": A1_Q, "choices": ch[::-1]}]}
    return req, ok_vals, group, img


def a2_request(src, r, rings):
    img = wrist_img(src, r, rings=rings)
    text, q = (A2_RINGS_TEXT, A2_RINGS_Q) if rings else (A2_SINGLE_TEXT, A2_SINGLE_Q)
    req = {"model": E.MODEL,
           "input": [{"role": "user", "content": [{"type": "input_text", "text": text},
                                                  {"type": "input_image", "image_url": img_url(img), "detail": "auto"}]}],
           "questions": [{"type": "predicate", "name": "fills", "instructions": q}]}
    return req, img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    fr = frames()
    jobs = []
    for src, r, G in fr:
        jobs.append(("A1", src, r, G, False))
        if src == GRID:
            jobs.append(("A1_mirror", src, r, G, True))
        t = a2_truth(r, G)
        if t is not None and r["hand_state"] == "open":
            jobs.append(("A2_single", src, r, G, t))
            jobs.append(("A2_rings", src, r, G, t))
    print(f"帧 {len(fr)}，请求 {len(jobs)}：{dict(Counter(j[0] for j in jobs))}")
    print("A1 标签：", dict(Counter(a1_label(s, r, G)[1] for s, r, G in fr)))
    print("A2 真值：", dict(Counter(j[4] for j in jobs if j[0] == "A2_single")))
    if args.dry:
        out = HERE / "eval_runs" / "_preview"
        out.mkdir(parents=True, exist_ok=True)
        s, r, G = fr[0]
        cv2.imwrite(str(out / "a1_grid0.jpg"), a1_request(s, r, G, False)[3])
        cv2.imwrite(str(out / "a2_rings.jpg"), a2_request(s, r, True)[1])
        print("预览 →", out)
        return 0

    from openai import OpenAI
    client = OpenAI(api_key=E.KEY_FILE.read_text().strip(), max_retries=0)
    throttle = E.Throttle(200)
    run_dir = HERE / "eval_runs" / ("near_" + time.strftime("%Y%m%d_%H%M%S"))
    (run_dir / "responses").mkdir(parents=True)
    results = []

    def work(kind, src, r, G, extra):
        if kind.startswith("A1"):
            req, ok_vals, group, _ = a1_request(src, r, G, extra)
        else:
            req, _ = a2_request(src, r, kind.endswith("rings"))
            ok_vals, group = None, extra
        res = E.call(client, req, throttle)
        (run_dir / "responses" / f"{kind}_{src.name}_{int(r['n']):04d}.json").write_text(
            json.dumps(res, ensure_ascii=False), encoding="utf-8")
        a = res["answers"]
        out = {"kind": kind, "src": src.name, "n": int(r["n"]), "phase": r["phase"], "group": group,
               "dz": float(r["p"][2] - G[2]), "off": float(np.linalg.norm((G - r["p"])[:2])), "ms": res["ms"]}
        if kind.startswith("A1"):
            out.update(ok=ok_vals, a=a.get("a", {}).get("choice"), b=a.get("b", {}).get("choice"))
        else:
            out.update(truth=extra, p=a.get("fills", {}).get("probability"))
        return out

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"\n  ✗ {type(e).__name__}: {e}")
            print(f"\r  {i}/{len(jobs)}", end="", flush=True)
    print()
    (run_dir / "results.json").write_text(json.dumps(
        [{**r, "ok": sorted(r["ok"]) if r.get("ok") else None} for r in results], ensure_ascii=False, indent=0),
        encoding="utf-8")

    lines = [f"# 近处小实验 {run_dir.name}"]
    for kind in ("A1", "A1_mirror"):
        rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        acc_a = np.mean([r["a"] in r["ok"] for r in rs])
        acc_b = np.mean([r["b"] in r["ok"] for r in rs])
        both = np.mean([r["a"] == r["b"] and r["a"] in r["ok"] for r in rs])
        lines.append(f"\n## {kind}（{len(rs)} 帧）：正序 {acc_a:.0%}、倒序 {acc_b:.0%}、两遍一致且对 {both:.0%}")
        by = defaultdict(list)
        for r in rs:
            key = ("网格" if r["src"] == GRID.name else "回放") + f" {r['group']}"
            by[key].append(r)
        for k, v in sorted(by.items()):
            lines.append(f"  - {k}（{len(v)}）对 {np.mean([r['a'] in r['ok'] for r in v]):.0%}；答成 "
                         + "、".join(f"{a}×{c}" for a, c in Counter(r["a"] for r in v).most_common(3)))
        offb = defaultdict(list)
        for r in rs:
            offb[int(round(r["off"] / 5) * 5)].append(r["a"] in r["ok"])
        lines.append("  - 按水平偏移：" + "  ".join(f"{k}mm {np.mean(v):.0%}({len(v)})" for k, v in sorted(offb.items())))
    for kind in ("A2_single", "A2_rings"):
        rs = [r for r in results if r["kind"] == kind and r["p"] is not None]
        if not rs:
            continue
        pos = [r["p"] for r in rs if r["truth"]]
        neg = [r for r in rs if not r["truth"]]
        fp = [r for r in neg if r["p"] >= 0.5]
        lines.append(f"\n## {kind}：该合手 {len(pos)} 帧判是 {sum(p >= 0.5 for p in pos)}（概率中位 {np.median(pos):.2f}）；"
                     f"不该合手 {len(neg)} 帧误报 {len(fp)}（概率中位 {np.median([r['p'] for r in neg]):.2f}）")
        fpz = Counter(int(round(r["dz"] / 10) * 10) for r in fp)
        lines.append("  - 误报按高度：" + "  ".join(f"+{k}mm×{v}" for k, v in sorted(fpz.items())))
    lines.append(f"\n共 {len(results)} 个请求")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("→", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
