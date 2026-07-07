# 【功能描述】Phase 0：从设计要求合成初始 T2I prompt
# 【输入】design_requirement、case_hint
# 【输出】合成后的 prompt 文本

from __future__ import annotations

from typing import Any

from promptopt.cases import build_synthesize_fields
from promptopt.clients.api_core_bridge import llm_chat_sync
from promptopt.templates import fill_prompt


def synthesize_initial_prompt(
    design_requirement: str,
    *,
    case_hint: str = "",
) -> str:
    user_content = fill_prompt("phase0_synthesize", {
        "design_requirement": design_requirement,
        "case_hint": case_hint or "无额外补充，仅依据设计要求发挥。",
    })
    messages = [{"role": "user", "content": user_content}]
    text, err = llm_chat_sync(messages)
    if err:
        raise RuntimeError(f"合成初始 prompt 失败: {err}")
    prompt = (text or "").strip()
    if not prompt:
        raise RuntimeError("合成初始 prompt 返回为空")
    return prompt + "\n"


def synthesize_from_kv_case(case: dict[str, Any]) -> str:
    fields = build_synthesize_fields(case)
    return synthesize_initial_prompt(
        fields["design_requirement"],
        case_hint=fields["case_hint"],
    )
