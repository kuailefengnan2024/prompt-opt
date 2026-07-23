# 【功能描述】T2I prompt 优化引擎默认参数 SSOT
# 【输入】scripts/run_t2i.py 顶部覆盖
# 【输出】DEFAULT_* 常量供 runner / clients 引用

from __future__ import annotations

# ── 任务输入 ──────────────────────────────────────────────────────────────
DEFAULT_CATEGORY: str = "3d"
DEFAULT_PROMPT_LIBRARY: str = "data/kv_synth_prompts_100_only.json"

# ── 优化循环 ──────────────────────────────────────────────────────────────
DEFAULT_MAX_ROUNDS: int = 8
DEFAULT_TRAIN_RUNS: int = 2
DEFAULT_GATE_RUNS: int = 2
DEFAULT_EDIT_BUDGET: int = 6
DEFAULT_HARD_THRESHOLD: float = 65.0
DEFAULT_MIN_ENSEMBLE_CONFIDENCE: float = 0.5
DEFAULT_REASON_MIN_DIM_CONFIDENCE: float = 0.833

# ── api-core 模型 ───────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER: str = "doubao21pro"
DEFAULT_IMAGE_PROVIDER: str = "seedream_4_5"
DEFAULT_IMAGE_SIZE: str = "2999x1687"
DEFAULT_ORIENTATION_MODE: int = 1

# ── aesthetic-core 单系审美 ─────────────────────────────────────────────────
DEFAULT_AESTHETIC_ENSEMBLE: dict[str, str] = {
    "doubao": "doubao_seed_2_0_pro_vision",
}

# ── Reflect ─────────────────────────────────────────────────────────────────
DEFAULT_MINIBATCH_SIZE: int = 8
DEFAULT_ANALYST_WORKERS: int = 4
DEFAULT_MERGE_BATCH_SIZE: int = 4
DEFAULT_SEED: int = 42

# ── Meta prompt（跨轮记忆）────────────────────────────────────────────────
DEFAULT_USE_META_PROMPT: bool = True
DEFAULT_META_PROMPT_MAX_CHARS: int = 1500

# ── 产物 ────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_ROOT: str = "outputs"
