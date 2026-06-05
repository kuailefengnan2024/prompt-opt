"""【功能描述】优化器侧 meta prompt 记忆（跨 epoch）；由相邻 epoch prompt 比较提炼，不写入 T2I prompt 正文，仅供后续 merge/rank 参考。

【输入】prev_prompt、curr_prompt、comparison_pairs、prev_meta_prompt_content 及可选 system_prompt。

【输出】format_meta_prompt_context 返回 prompt 上下文块；run_meta_prompt 返回含 meta_prompt_content 的 dict 或 None。
"""
from __future__ import annotations

import traceback

from skillopt.model import chat_optimizer
from skillopt.optimizer.slow_update import format_comparison_text
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


def format_meta_prompt_context(meta_prompt_content: str) -> str:
    """将优化器记忆渲染为可用于 prompt 的上下文块。"""
    content = (meta_prompt_content or "").strip()
    if not content:
        return ""
    return (
        "## Optimizer Meta Prompt\n"
        "This is optimizer-side memory distilled from prior epoch transitions in "
        "this environment. Use it to improve how you propose, merge, and rank "
        "prompt edits. Prefer it when the current evidence is ambiguous, but do "
        "not force it if the current trajectories clearly contradict it.\n\n"
        f"{content}"
    )


def run_meta_prompt(
    prev_prompt: str,
    curr_prompt: str,
    comparison_pairs: list[dict],
    *,
    prev_meta_prompt_content: str = "",
    system_prompt: str | None = None,
) -> dict | None:
    """根据相邻 epoch 生成更新后的优化器侧 meta prompt 记忆。"""
    actual_system = system_prompt if system_prompt is not None else load_prompt("meta_prompt")

    prev_prompt_display = prev_prompt
    if len(prev_prompt_display) > 6000:
        prev_prompt_display = prev_prompt_display[:6000] + "\n...[truncated]..."

    curr_prompt_display = curr_prompt
    if len(curr_prompt_display) > 6000:
        curr_prompt_display = curr_prompt_display[:6000] + "\n...[truncated]..."

    prev_meta_section = (
        prev_meta_prompt_content.strip()
        if prev_meta_prompt_content and prev_meta_prompt_content.strip()
        else "(No previous optimizer meta prompt — this is the first update.)"
    )

    comparison_text = format_comparison_text(comparison_pairs)
    user = (
        f"## Previous Epoch Last-Step Prompt\n{prev_prompt_display}\n\n"
        f"## Current Epoch Last-Step Prompt\n{curr_prompt_display}\n\n"
        f"## Previous Optimizer Meta Prompt\n"
        f"The following optimizer memory was available during the current epoch. "
        f"Reflect on whether it improved or harmed the quality of edits.\n\n"
        f"{prev_meta_section}\n\n"
        f"## Longitudinal Comparison (same tasks, two last-step prompts)\n"
        f"{comparison_text}"
    )

    try:
        response, _ = chat_optimizer(
            system=actual_system,
            user=user,
            max_completion_tokens=3072,
            retries=3,
            stage="meta_prompt",
        )
        result = extract_json(response)
        if result and result.get("meta_prompt_content"):
            return {
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_prompt_content": str(result["meta_prompt_content"]).strip(),
            }
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None
