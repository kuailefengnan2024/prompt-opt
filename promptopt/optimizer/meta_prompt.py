# 【功能描述】T2I 跨轮优化器记忆：格式化注入 + LLM 滚动更新
# 【输入】轮次摘要、上轮记忆、设计要求
# 【输出】meta_prompt_content 字符串；format_meta_prompt_context 供 Reflect 模板注入

from __future__ import annotations

import json
import traceback
from typing import Any

from promptopt.model import chat_optimizer
from promptopt.design_requirement import format_design_requirement_section
from promptopt.templates import fill_prompt, has_prompt
from promptopt.utils import extract_json

DEFAULT_META_MAX_CHARS = 1500


def format_meta_prompt_context(meta_prompt_content: str) -> str:
    content = (meta_prompt_content or "").strip()
    if not content:
        return ""
    return f"## 优化器记忆\n{content}\n\n"


def _clip(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _summarize_edits(patch: dict[str, Any] | None, limit: int = 6) -> str:
    if not patch:
        return "（无 patch）"
    reasoning = _clip(str(patch.get("reasoning", "")), 300)
    edits = patch.get("edits") or []
    lines = [f"reasoning: {reasoning}"] if reasoning else []
    for i, e in enumerate(edits[:limit]):
        if not isinstance(e, dict):
            continue
        op = e.get("op", "?")
        target = _clip(str(e.get("target", "")), 80)
        content = _clip(str(e.get("content", "")), 120)
        lines.append(f"  edit[{i}] {op} target={target!r} → {content!r}")
    if len(edits) > limit:
        lines.append(f"  ... +{len(edits) - limit} more edits")
    return "\n".join(lines) if lines else "（空 patch）"


def _summarize_rollout_scores(results: list[dict[str, Any]] | None) -> str:
    if not results:
        return "n/a"
    scores = [float(r.get("final_score", r.get("soft", 0) * 100) or 0) for r in results]
    if not scores:
        return "n/a"
    avg = sum(scores) / len(scores)
    reasons = [_clip(str(r.get("fail_reason", "")), 120) for r in results if r.get("fail_reason")]
    reason_part = f"; fail_reasons={reasons[:2]}" if reasons else ""
    return f"avg_final={avg:.2f} (n={len(scores)}){reason_part}"


def build_round_digest(
    *,
    round_idx: int,
    action: str,
    rollout_soft: float | None = None,
    gate_soft: float | None = None,
    current_score: float | None = None,
    best_score: float | None = None,
    ranked_patch: dict[str, Any] | None = None,
    merged_patch: dict[str, Any] | None = None,
    rollout_results: list[dict[str, Any]] | None = None,
    gate_results: list[dict[str, Any]] | None = None,
    extra_note: str = "",
) -> str:
    """将单轮结果压缩为 meta 更新用的文本块。"""
    lines = [
        f"## Round {round_idx}",
        f"- action: {action}",
    ]
    if rollout_soft is not None:
        lines.append(f"- rollout_soft: {rollout_soft:.4f} ({_summarize_rollout_scores(rollout_results)})")
    if gate_soft is not None:
        lines.append(f"- gate_soft: {gate_soft:.4f} ({_summarize_rollout_scores(gate_results)})")
    if current_score is not None:
        lines.append(f"- current_score: {current_score:.4f}")
    if best_score is not None:
        lines.append(f"- best_score: {best_score:.4f}")
    if extra_note.strip():
        lines.append(f"- note: {extra_note.strip()}")
    lines.append("- patch_tried:")
    lines.append(_summarize_edits(merged_patch or ranked_patch))
    return "\n".join(lines)


def build_history_digest(history: list[dict[str, Any]], max_rounds: int = 4) -> str:
    """拼接最近若干轮摘要，供 meta LLM 总览。"""
    if not history:
        return "（尚无历史轮次）"
    chunks: list[str] = []
    for rec in history[-max_rounds:]:
        if not isinstance(rec, dict):
            continue
        chunks.append(
            build_round_digest(
                round_idx=int(rec.get("round", 0) or 0),
                action=str(rec.get("action", "unknown")),
                rollout_soft=rec.get("rollout_soft"),
                gate_soft=rec.get("gate_soft"),
                current_score=rec.get("current_score"),
                best_score=rec.get("best_score"),
                ranked_patch=rec.get("merged_patch") or rec.get("ranked_patch"),
                extra_note=str(rec.get("meta_note", "") or ""),
            )
        )
    return "\n\n".join(chunks)


def run_meta_prompt_update(
    *,
    prompt_excerpt: str,
    design_requirement: str = "",
    previous_meta: str,
    round_digest: str,
    max_chars: int = DEFAULT_META_MAX_CHARS,
) -> dict[str, Any] | None:
    """调用 optimizer LLM 滚动更新跨轮记忆。"""
    if not has_prompt("meta_prompt"):
        raise FileNotFoundError("缺少模板 promptopt/prompts/meta_prompt.md")

    user = fill_prompt("meta_prompt", {
        "prompt_excerpt": _clip(prompt_excerpt, 800),
        "design_requirement_section": format_design_requirement_section(design_requirement).strip(),
        "previous_meta": previous_meta.strip() or "（空）",
        "round_digest": round_digest.strip() or "（无新轮次信息）",
        "max_chars": str(max_chars),
    })
    try:
        response, _ = chat_optimizer(
            system="",
            user=user,
            max_completion_tokens=2048,
            retries=3,
            stage="meta_prompt",
        )
        result = extract_json(response)
        if not result:
            return None
        content = str(result.get("meta_prompt_content", "") or "").strip()
        if len(content) > max_chars:
            content = content[: max_chars - 3] + "..."
        return {
            "reasoning": str(result.get("reasoning", "") or "").strip(),
            "meta_prompt_content": content,
        }
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None
