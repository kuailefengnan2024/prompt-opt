#!/usr/bin/env python3
# 【功能描述】T2I prompt 优化入口：随机库内 prompt → Reflect（仅改画面构图）→ outputs/
# 【输入】本文件顶部 ★ 高频配置；data/kv_synth_prompts_100_only.json
# 【输出】outputs/<run_id>/ 下 case.json、initial/、rounds/、best/、summary.json

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# ★ 高频配置（改这里）
# =============================================================================

# ── Case / 输入 ──────────────────────────────────────────────────────────────
CASE_SEED = None  # None=每次随机；设整数可复现同一条库内 prompt
CATEGORY = "3d"   # 品类：3d | graphic | illustration
PROMPT_LIBRARY = "data/kv_synth_prompts_100_only.json"  # 字符串数组，入参只用 prompt

# ── 训练循环 ────────────────────────────────────────────────────────────────
MAX_ROUNDS = 8    # 正式跑满 N 轮，取 history 中 best_score 最高 prompt
TRAIN_RUNS = 4    # 每轮 Reflect 前 rollout 张数（多次采样取均分）
GATE_RUNS = 2     # Gate 验证张数（多次采样取均分）
EDIT_BUDGET = 4   # 每步最多几条局部 patch（学习率）
SEED = 42         # Reflect minibatch shuffle 等可复现种子

# ── 审美打分 ────────────────────────────────────────────────────────────────
HARD_THRESHOLD = 65.0              # final_score >= 此值 → hard=1
MIN_ENSEMBLE_CONFIDENCE = 0.5      # ensemble 置信度过低则标注 unreliable（仍参与打分）
REASON_MIN_DIM_CONFIDENCE = 0.833  # 子维度 reason 过滤阈值，供 trajectory 展示
AESTHETIC_ENSEMBLE = {             # 审美单系（仅 doubao）
    "doubao": "doubao_seed_2_0_pro_vision",
}

# ── 模型 Provider ───────────────────────────────────────────────────────────
LLM_PROVIDER = "doubao21pro"       # api-core 注册名
IMAGE_PROVIDER = "seedream_4_5"
IMAGE_SIZE = "2999x1687"

# ── Reflect 并行 ────────────────────────────────────────────────────────────
MINIBATCH_SIZE = 8
ANALYST_WORKERS = 4
MERGE_BATCH_SIZE = 4

# ── Meta 记忆 ───────────────────────────────────────────────────────────────
USE_META_PROMPT = True          # 每轮 Gate 后滚动更新，下轮 Reflect 注入
META_PROMPT_MAX_CHARS = 1500

# ── 产物 / 调试 ─────────────────────────────────────────────────────────────
OUTPUT_ROOT = "outputs"  # 每次运行子目录 outputs/<timestamp>/
SAVE_DEBUG = False       # True：保留 _work 中间产物
OPEN_REPORT = True       # 结束后用默认浏览器打开 report.html

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

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = str(_PROJECT_ROOT / OUTPUT_ROOT / f"t2i_{run_id}")
    os.makedirs(out_root, exist_ok=True)

    case = pick_random_synth_prompt(
        seed=CASE_SEED,
        path=_PROJECT_ROOT / PROMPT_LIBRARY,
    )
    print(f"随机库内 prompt: index={case['index']} chars={len(case['prompt'])}")

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

    print(f"输出目录: {out_root}")
    print(f"轮数: {MAX_ROUNDS} | train_runs={TRAIN_RUNS} | gate_runs={GATE_RUNS} | meta_prompt={USE_META_PROMPT}")
    summary = run_t2i_optimize(cfg)
    print("\n" + "=" * 60)
    print("完成")
    best = summary.get("best") or {}
    print(f"  case_index: {summary.get('case_index')}")
    print(f"  best_score: {best.get('score')}")
    print(f"  best_step:  {best.get('step')}")
    print(f"  产物:       {out_root}")
    print(f"  报告:       {summary.get('report', out_root + '/report.html')}")
    print("  交付物:     initial/  best/  summary.json  report.html")
    print("=" * 60)

    report_path = summary.get("report") or str(Path(out_root) / "report.html")
    if OPEN_REPORT and Path(report_path).is_file():
        import webbrowser
        webbrowser.open(Path(report_path).resolve().as_uri())
        print(f"  已打开报告: {report_path}")


if __name__ == "__main__":
    main()
