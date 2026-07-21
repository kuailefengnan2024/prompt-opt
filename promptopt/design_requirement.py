# 【功能描述】解析并格式化「原始设计要求」，供审美打分与 Reflect/Merge/Rank 注入
# 【输入】initial_prompt、可选 case_meta（design_requirement / main_title 等）
# 【输出】resolve_design_requirement 字符串；format_design_requirement_section 模板片段

from __future__ import annotations

import re
from typing import Any

_TITLE_MAIN_RE = re.compile(r'主标题为["“]([^"”]+)["”]')
_TITLE_SUB_RE = re.compile(r'副标题为["“]([^"”]+)["”]')


def resolve_design_requirement(initial_prompt: str, case_meta: dict[str, Any] | None = None) -> str:
    """从 case_meta 或 initial_prompt 提取结构化设计要求。"""
    meta = case_meta or {}

    direct = str(meta.get("design_requirement") or "").strip()
    if direct:
        return direct

    parts: list[str] = []
    main = str(meta.get("main_title") or "").strip()
    sub = str(meta.get("sub_title") or "").strip()
    other = str(meta.get("other_requirements") or "").strip()
    imagery = str(meta.get("imagery_requirement") or "").strip()

    if main:
        parts.append(f"主标题：{main}")
    if sub:
        parts.append(f"副标题：{sub}")
    if imagery and imagery not in other:
        parts.append(f"主体意象：{imagery}")
    if other:
        parts.append(f"其他要求：{other}")
    if parts:
        return "\n".join(parts)

    prompt = (initial_prompt or "").strip()
    if not prompt:
        return ""

    m_main = _TITLE_MAIN_RE.search(prompt)
    m_sub = _TITLE_SUB_RE.search(prompt)
    if m_main:
        parts.append(f"主标题：{m_main.group(1)}")
    if m_sub:
        parts.append(f"副标题：{m_sub.group(1)}")
    if parts:
        return "\n".join(parts)

    return prompt[:800] + ("..." if len(prompt) > 800 else "")


def format_design_requirement_section(design_requirement: str) -> str:
    """供 Reflect / Merge / Rank / Meta 模板注入。"""
    content = (design_requirement or "").strip()
    if not content:
        return ""
    return f"## 约束锚点（设计要求 · 不得偏离）\n{content}\n\n"
