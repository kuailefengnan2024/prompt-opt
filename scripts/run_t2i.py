#!/usr/bin/env python3
# 【功能描述】T2I prompt 优化入口：随机 KV case → Phase0 合成 → 8 轮 Reflect → outputs/
# 【输入】本文件顶部 ★ 高频配置
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

# True：从 data/user_cases.json 随机选一条 KV 设计要求；False：用下方 DESIGN_REQUIREMENT
USE_RANDOM_KV_CASE = True

# 仅当 USE_RANDOM_KV_CASE=False 时生效
DESIGN_REQUIREMENT = (
    "主标题：夏日音乐节\n"
    "副标题：2026 · 北京站\n"
    "其他要求：蓝紫霓虹、横版 16:9、画面简洁、主标题区留白干净。"
)

# 品类：3d | graphic | illustration
CATEGORY = "3d"

# 正式跑满 8 轮，取 history 中 best_score 最高 prompt
MAX_ROUNDS = 8

# 每轮 rollout / gate 各出几张图（无 seed，多次采样取均分）
TRAIN_RUNS = 2
GATE_RUNS = 2

# 每步最多几条局部 patch（学习率）
EDIT_BUDGET = 3

# 审美及格线：final_score >= 此值 → hard=1
HARD_THRESHOLD = 65.0

# ensemble 整体置信度过低则标注 unreliable（仍参与打分）
MIN_ENSEMBLE_CONFIDENCE = 0.5

# 子维度 reason 已由 aesthetic-core 按 0.833 过滤；此处供 trajectory 展示
REASON_MIN_DIM_CONFIDENCE = 0.833

# api-core 注册名
LLM_PROVIDER = "doubao21pro"
IMAGE_PROVIDER = "seedream_4_5"
IMAGE_SIZE = "2999x1687"

# 审美单系（仅 doubao）
AESTHETIC_ENSEMBLE = {
    "doubao": "doubao_seed_2_0_pro_vision",
}

# Reflect 并行
MINIBATCH_SIZE = 8
ANALYST_WORKERS = 4
MERGE_BATCH_SIZE = 4
SEED = 42

# None=每次随机选 case；设整数可复现同一 case
CASE_SEED = None

# 产物根目录（每次运行子目录 outputs/<timestamp>/）
OUTPUT_ROOT = "outputs"

# 中间产物默认不落盘；排错时可改 True 保留 rounds/ 下 patch 等细节
SAVE_DEBUG = False

# 跨轮优化器记忆：每轮 Gate 后 LLM 滚动更新，下轮 Reflect 注入
USE_META_PROMPT = True
META_PROMPT_MAX_CHARS = 1500

# 训练结束后用默认浏览器打开 report.html
OPEN_REPORT = True

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
    from promptopt.cases import pick_random_kv_case

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = str(_PROJECT_ROOT / OUTPUT_ROOT / f"t2i_{run_id}")
    os.makedirs(out_root, exist_ok=True)

    kv_case = None
    design_requirement = DESIGN_REQUIREMENT
    if USE_RANDOM_KV_CASE:
        kv_case = pick_random_kv_case(seed=CASE_SEED)
        design_requirement = kv_case["design_requirement"]
        print(f"随机 KV case: {kv_case.get('case_id')} — {kv_case.get('main_title')}")

    cfg = T2IRunConfig(
        design_requirement=design_requirement,
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
        kv_case=kv_case,
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
    print(f"  case:       {summary.get('case_id')} {summary.get('main_title')}")
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
