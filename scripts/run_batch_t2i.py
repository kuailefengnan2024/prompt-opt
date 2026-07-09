#!/usr/bin/env python3
# 【功能描述】批量 T2I 优化：按预设 CASE_SEEDS 依次从库内取 prompt 跑满链路
# 【输入】本文件顶部 ★ 配置；data/kv_synth_prompts_100_only.json
# 【输出】outputs/t2i_<ts>/ × N；outputs/batch_<ts>/manifest.json

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# ★ 批量配置（改这里）
# =============================================================================

# 10 个 case 的随机种子（对应库内不同 prompt）
CASE_SEEDS = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
PROMPT_LIBRARY = "data/kv_synth_prompts_100_only.json"

CATEGORY = "3d"
MAX_ROUNDS = 8
TRAIN_RUNS = 2
GATE_RUNS = 2
EDIT_BUDGET = 3
HARD_THRESHOLD = 65.0
MIN_ENSEMBLE_CONFIDENCE = 0.5
REASON_MIN_DIM_CONFIDENCE = 0.833
LLM_PROVIDER = "doubao21pro"
IMAGE_PROVIDER = "seedream_4_5"
IMAGE_SIZE = "2999x1687"
AESTHETIC_ENSEMBLE = {"doubao": "doubao_seed_2_0_pro_vision"}
MINIBATCH_SIZE = 8
ANALYST_WORKERS = 4
MERGE_BATCH_SIZE = 4
SEED = 42
OUTPUT_ROOT = "outputs"
SAVE_DEBUG = False
USE_META_PROMPT = True
META_PROMPT_MAX_CHARS = 1500

# 批量跑时不每个都弹浏览器；仅最后一个完成后打开
OPEN_REPORT_EACH = False
OPEN_REPORT_LAST = True

# =============================================================================

_AESTHETIC_ROOT = Path(r"D:\kv-generator\engines\aesthetic-core")
if _AESTHETIC_ROOT.exists() and str(_AESTHETIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AESTHETIC_ROOT))


def _load_dotenv() -> None:
    for path in (_PROJECT_ROOT / ".env", Path(r"D:\api-core\.env"), Path(r"D:\kv-generator\.env")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith("export "):
                key = key[7:].strip()
            os.environ.setdefault(key, val)


def main() -> None:
    _load_dotenv()
    from promptopt.engine.runner import T2IRunConfig, run_t2i_optimize
    from promptopt.cases import pick_random_synth_prompt

    batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = _PROJECT_ROOT / OUTPUT_ROOT / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    lib_path = _PROJECT_ROOT / PROMPT_LIBRARY

    manifest: list[dict] = []
    total = len(CASE_SEEDS)
    t_batch = time.time()

    print(f"批量任务启动: {total} 个 case")
    print(f"清单目录: {batch_dir}")

    last_report = ""

    for i, case_seed in enumerate(CASE_SEEDS, 1):
        case = pick_random_synth_prompt(seed=case_seed, path=lib_path)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = str(_PROJECT_ROOT / OUTPUT_ROOT / f"t2i_{run_id}")
        os.makedirs(out_root, exist_ok=True)

        print("\n" + "#" * 70)
        print(f"[{i}/{total}] case_seed={case_seed} → index={case['index']} chars={len(case['prompt'])}")
        print(f"输出: {out_root}")
        print("#" * 70)

        t0 = time.time()
        cfg = T2IRunConfig(
            initial_prompt=case["prompt"],
            category=CATEGORY,
            max_rounds=MAX_ROUNDS,
            train_runs=TRAIN_RUNS,
            gate_runs=GATE_RUNS,
            edit_budget=EDIT_BUDGET,
            hard_threshold=HARD_THRESHOLD,
            min_ensemble_confidence=MIN_ENSEMBLE_CONFIDENCE,
            reason_min_dim_confidence=REASON_MIN_DIM_CONFIDENCE,
            llm_provider=LLM_PROVIDER,
            image_provider=IMAGE_PROVIDER,
            image_size=IMAGE_SIZE,
            aesthetic_ensemble=AESTHETIC_ENSEMBLE,
            minibatch_size=MINIBATCH_SIZE,
            analyst_workers=ANALYST_WORKERS,
            merge_batch_size=MERGE_BATCH_SIZE,
            seed=SEED,
            out_root=out_root,
            case_meta=case,
            save_debug=SAVE_DEBUG,
            use_meta_prompt=USE_META_PROMPT,
            meta_prompt_max_chars=META_PROMPT_MAX_CHARS,
        )

        try:
            summary = run_t2i_optimize(cfg)
            elapsed = round(time.time() - t0, 1)
            best = summary.get("best") or {}
            report_path = summary.get("report") or str(Path(out_root) / "report.html")
            last_report = report_path

            row = {
                "index": i,
                "case_seed": case_seed,
                "case_index": summary.get("case_index"),
                "out_root": out_root,
                "report": report_path,
                "initial_score": (summary.get("initial") or {}).get("final_score"),
                "best_score": best.get("final_score"),
                "best_step": best.get("step"),
                "elapsed_s": elapsed,
                "status": "ok",
            }
            manifest.append(row)
            print(f"  ✓ 完成 best={best.get('final_score')} step={best.get('step')} ({elapsed}s)")

            if OPEN_REPORT_EACH and Path(report_path).is_file():
                webbrowser.open(Path(report_path).resolve().as_uri())

        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 1)
            manifest.append({
                "index": i,
                "case_seed": case_seed,
                "case_index": case.get("index"),
                "out_root": out_root,
                "elapsed_s": elapsed,
                "status": "error",
                "error": str(exc),
            })
            print(f"  ✗ 失败: {exc}")

        with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": batch_id,
                "total": total,
                "completed": len(manifest),
                "runs": manifest,
            }, f, ensure_ascii=False, indent=2)

    batch_elapsed = round(time.time() - t_batch, 1)
    print("\n" + "=" * 70)
    print(f"批量完成 {len(manifest)}/{total}，总耗时 {batch_elapsed}s")
    print(f"清单: {batch_dir / 'manifest.json'}")
    for row in manifest:
        st = row.get("status", "?")
        print(f"  [{row.get('index')}] case_index={row.get('case_index')} {st} best={row.get('best_score')}")
    print("=" * 70)

    if OPEN_REPORT_LAST and last_report and Path(last_report).is_file():
        webbrowser.open(Path(last_report).resolve().as_uri())
        print(f"已打开最后一个报告: {last_report}")


if __name__ == "__main__":
    main()
