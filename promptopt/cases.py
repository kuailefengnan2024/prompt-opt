# 【功能描述】从 KV user_cases 随机选取设计要求 case
# 【输入】data/user_cases.json、可选 seed
# 【输出】case 字典及 Phase0 字段 design_requirement / case_hint

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CASES_PATH = _PROJECT_ROOT / "data" / "user_cases.json"
_KV_CASES_PATH = Path(r"D:\kv-generator\backend\assets_user\user_cases.json")


def _resolve_cases_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    if _DEFAULT_CASES_PATH.is_file():
        return _DEFAULT_CASES_PATH
    if _KV_CASES_PATH.is_file():
        return _KV_CASES_PATH
    raise FileNotFoundError(
        f"未找到 user_cases.json，请复制至 {_DEFAULT_CASES_PATH} 或保留 KV 路径 {_KV_CASES_PATH}"
    )


def load_kv_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    cases_path = _resolve_cases_path(path)
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"user_cases 格式无效: {cases_path}")
    return cases


def pick_random_kv_case(*, seed: int | None = None, path: str | Path | None = None) -> dict[str, Any]:
    cases = load_kv_cases(path)
    rng = random.Random(seed)
    return dict(rng.choice(cases))


def build_synthesize_fields(case: dict[str, Any]) -> dict[str, str]:
    subjects = case.get("main_subjects") or []
    imagery = (case.get("imagery_requirement") or "").strip()
    style = (case.get("other_requirements") or "").strip()
    if not style and "\n其他要求：" in case.get("design_requirement", ""):
        style = case["design_requirement"].split("\n其他要求：", 1)[-1].strip()

    hint_parts: list[str] = []
    if imagery:
        hint_parts.append(f"意象参考：{imagery}")
    if subjects:
        hint_parts.append(f"主体意象：{', '.join(subjects)}")
    if style:
        hint_parts.append(f"风格要求：{style}")

    return {
        "design_requirement": case["design_requirement"].strip(),
        "case_hint": "\n".join(hint_parts) or "无",
    }
