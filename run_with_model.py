#!/usr/bin/env python3
"""用更大的模型跑 auto_loop.py（原样，不改一行）：只把「问模型」从决策接口换成通用接口（Responses + 只能选一个值的 JSON 格式）。

10-14：决策接口只接受 gpt-6-luna（astra / sol / 6.1-sol 都 404），但这几个模型走通用接口都能用（一次 4~5 秒）。
题目、选项、图片都和原来一样：auto_loop 生成的请求里每道题有正序、倒序两份选项，这里两份同时各问一次，
再按原来的规则（两次一致才执行、不一致重问、3 轮后多数票）处理。

    python run_with_model.py --model gpt-6.1-sol           # 真跑（会要你输入 yes）
    python run_with_model.py --model gpt-6.1-sol --selftest   # 只用存下的图问 2 道题，不连硬件
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_decisions as E  # noqa: E402
import closed_loop as CL  # noqa: E402


def make_responses_decider(model, log_dir):
    from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
    client = OpenAI(api_key=E.KEY_FILE.read_text().strip(), max_retries=0)
    counter = {"n": 0}

    def ask_one(content, q):
        vals = [c["value"] for c in q["choices"]]
        opts = "\n".join(f"- {c['value']}: {c['description']}" for c in q["choices"])
        parts = list(content) + [{"type": "input_text",
                                  "text": f"\nQuestion: {q['instructions']}\nOptions (answer with exactly one value):\n{opts}"}]
        for w in (3, 6, 12, 24, None):
            try:
                r = client.responses.create(
                    model=model, input=[{"role": "user", "content": parts}],
                    text={"format": {"type": "json_schema", "name": "answer", "strict": True,
                                     "schema": {"type": "object", "additionalProperties": False,
                                                "properties": {"choice": {"type": "string", "enum": vals}},
                                                "required": ["choice"]}}})
                return json.loads(r.output_text)["choice"], r.usage.input_tokens
            except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as e:
                if w is None:
                    raise
                print(f"  接口暂时不可用（{type(e).__name__}），{w}s 后重试 ...", flush=True)
                time.sleep(w)

    def real(req, kind):
        content = req["input"][0]["content"]
        t = time.perf_counter()
        with ThreadPoolExecutor(len(req["questions"])) as ex:
            futs = {q["name"]: ex.submit(ask_one, content, q) for q in req["questions"]}
            res = {k: f.result() for k, f in futs.items()}
        out = {"ms": (time.perf_counter() - t) * 1000, "input_tokens": sum(v[1] for v in res.values()),
               "model": model, "answers": {k: {"choice": v[0]} for k, v in res.items()}}
        counter["n"] += 1
        if log_dir is not None:
            (Path(log_dir) / f"api_{counter['n']:03d}_{kind}.json").write_text(json.dumps(out, ensure_ascii=False))
        return out
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--selftest", action="store_true")
    args, rest = ap.parse_known_args()

    CL.make_decider = lambda a, log_dir: (make_responses_decider(args.model, log_dir) if not a.mock
                                          else _orig(a, log_dir))
    E.MODEL = args.model                                       # 只记在请求里，换接口后实际用 args.model

    if args.selftest:
        import glob
        import cv2
        import auto_loop as A
        dec = make_responses_decider(args.model, None)
        d = sorted(glob.glob(str(Path.home() / "atom_episodes/robotdecision_auto/20261014_093856")))[0]
        w = cv2.imread(f"{d}/030_wrist.jpg")
        side = cv2.imread(f"{d}/030_side.jpg")
        r, _ = A.ask_phase(dec, w)
        print("位置题：", r["answers"]["a"]["choice"], "/", r["answers"]["b"]["choice"], f"{r['ms'] / 1000:.1f}s")
        letters = {"A": "north", "B": "east", "C": "south", "D": "west"}
        r, _ = A.ask_ring(dec, w, side, letters)
        print("环内题：", r["answers"]["a"]["choice"], "/", r["answers"]["b"]["choice"], f"{r['ms'] / 1000:.1f}s")
        return 0

    import auto_loop as A
    print(f"auto_loop.py 原样运行，模型换成 {args.model}（通用接口）", flush=True)
    sys.argv = ["auto_loop.py"] + rest
    return A.main()


_orig = CL.make_decider

if __name__ == "__main__":
    sys.exit(main())
