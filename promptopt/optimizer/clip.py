"""【功能描述】Reflect 梯度裁剪 — 规则选取 top-L edits，替代 LLM Rank。

【输入】patch dict、max_edits、update_mode。

【输出】裁剪后的 Patch dict（含 clip_details）。
"""
from __future__ import annotations

from typing import Any

from promptopt.optimizer.update_modes import get_payload_items, normalize_update_mode, payload_key, payload_label


def _edit_sort_key(edit: dict[str, Any]) -> tuple[int, int, int]:
    """failure 优先 → support_count 降序 → merge_level 升序。"""
    source = str(edit.get("source_type") or "").lower()
    failure_rank = 0 if source == "failure" else 1
    support = int(edit.get("support_count") or 0)
    merge_level = int(edit.get("merge_level") or 0)
    return (failure_rank, -support, merge_level)


def _dedupe_edits_by_target(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同 target 保留优先级最高的一条。"""
    best_by_target: dict[str, dict[str, Any]] = {}
    no_target: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        target = str(edit.get("target") or "").strip()
        if not target:
            no_target.append(edit)
            continue
        prev = best_by_target.get(target)
        if prev is None or _edit_sort_key(edit) < _edit_sort_key(prev):
            best_by_target[target] = edit
    merged = list(best_by_target.values()) + no_target
    merged.sort(key=_edit_sort_key)
    return merged


def clip_edits(
    patch: dict,
    max_edits: int,
    update_mode: str = "patch",
) -> dict:
    """按规则裁剪 edits 至 budget 内；已在预算内则原样返回。"""
    update_mode = normalize_update_mode(update_mode)
    key = payload_key(update_mode)
    edits = [e for e in get_payload_items(patch, update_mode) if isinstance(e, dict)]
    if max_edits < 0 or len(edits) <= max_edits:
        return patch

    selected = _dedupe_edits_by_target(edits)[:max_edits]
    label = payload_label(update_mode)
    return {
        "reasoning": (
            f"{patch.get('reasoning', '')} "
            f"[rule-clipped: {len(edits)}→{len(selected)} {label}]"
        ).strip(),
        key: selected,
        "clip_details": {
            "method": "rule",
            "input_count": len(edits),
            "output_count": len(selected),
            "max_edits": max_edits,
        },
    }


def rank_and_select(
    patch: dict,
    max_edits: int,
    update_mode: str = "patch",
    **_ignored: Any,
) -> dict:
    """兼容旧调用签名；忽略 prompt / meta / design 参数。"""
    return clip_edits(patch, max_edits=max_edits, update_mode=update_mode)
