# 【功能描述】从合成 prompt 库随机选取初始文生图提示词
# 【输入】data/kv_synth_prompts_100_only.json（字符串数组）、可选 seed / path
# 【输出】含 index / prompt 的 case 字典

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPTS_PATH = _PROJECT_ROOT / "data" / "kv_synth_prompts_100_only.json"


def _resolve_prompts_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    if _DEFAULT_PROMPTS_PATH.is_file():
        return _DEFAULT_PROMPTS_PATH
    raise FileNotFoundError(f"未找到 prompt 库: {_DEFAULT_PROMPTS_PATH}")


def load_synth_prompts(path: str | Path | None = None) -> list[str]:
    prompts_path = _resolve_prompts_path(path)
    with open(prompts_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"prompt 库格式无效（须为非空字符串数组）: {prompts_path}")
    prompts = [str(p).strip() for p in data if str(p).strip()]
    if not prompts:
        raise ValueError(f"prompt 库无有效条目: {prompts_path}")
    return prompts


def pick_random_synth_prompt(
    *,
    seed: int | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """随机取一条初始 prompt，返回 {index, prompt, source}。"""
    prompts = load_synth_prompts(path)
    rng = random.Random(seed)
    index = rng.randrange(len(prompts))
    return {
        "index": index,
        "prompt": prompts[index],
        "source": str(_resolve_prompts_path(path)),
    }
