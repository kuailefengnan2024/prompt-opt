"""【功能描述】由优化器驱动的完整 prompt 重写，基于选中的 revise_suggestions 生成新 prompt 文档。

【输入】prompt_content、patch（含 suggestions）、可选 system_prompt、step_buffer_context、env 等。

【输出】含 new_prompt 与 change_summary 的结果 dict，失败时返回 None。
"""
from __future__ import annotations

import json

from promptopt.model import chat_optimizer
from promptopt.llm_templates import load_template
from promptopt.optimizer.update_modes import get_payload_items
from promptopt.utils import extract_json


def rewrite_prompt_from_suggestions(
    prompt_content: str,
    patch: dict,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    env: str | None = None,
    reasoning_effort: str | None = "high",
    max_completion_tokens: int = 64000,
) -> dict | None:
    suggestions = get_payload_items(patch, "rewrite_from_suggestions")
    if not suggestions:
        return None

    user = (
        f"## Current Prompt\n{prompt_content}\n\n"
        f"## Selected Revise Suggestions ({len(suggestions)} total)\n"
        f"{json.dumps(suggestions, ensure_ascii=False, indent=2)}\n\n"
    )
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    user += (
        "Rewrite the full prompt document so it integrates the selected suggestions. "
        "Return the complete new prompt in `new_prompt`."
    )

    actual_system = system_prompt if system_prompt is not None else load_template(
        "rewrite_prompt", env=env,
    )

    try:
        response, _ = chat_optimizer(
            system=actual_system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rewrite",
            reasoning_effort=reasoning_effort,
        )
        result = extract_json(response)
        if result and str(result.get("new_prompt", "")).strip():
            result["new_prompt"] = str(result["new_prompt"]).rstrip() + "\n"
            if "change_summary" not in result or not isinstance(result["change_summary"], list):
                result["change_summary"] = []
            return result
    except Exception:  # noqa: BLE001
        return None
    return None
