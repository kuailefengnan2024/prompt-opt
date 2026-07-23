# 【功能描述】解析并格式化「设计要求」约束锚点，供审美打分与 Reflect/Merge/Meta 注入
# 【输入】显式 design_requirement 字符串，或 case_meta 字段；不以全文 prompt 回退
# 【输出】resolve_design_requirement 短文本；format_design_requirement_section 模板片段

from __future__ import annotations

import re
from typing import Any

_TITLE_MAIN_RE = re.compile(r'主标题\s*[为:]?\s*["“]([^"”]+)["”]')
_TITLE_SUB_RE = re.compile(r'副标题\s*[为:]?\s*["“]([^"”]+)["”]')
_DECOR_RE = re.compile(r'(?:装饰文案|小字(?:装饰文案)?)\s*["“]([^"”]+)["”]')
_CORE_RE = re.compile(r"【核心特征】\s*(.*?)(?=【[^】]+】|\Z)", re.DOTALL)


def _clip(text: str, limit: int = 220) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def resolve_design_requirement(
    initial_prompt: str = "",
    case_meta: dict[str, Any] | None = None,
    *,
    design_requirement: str | None = None,
) -> str:
    """优先使用显式传入的设计要求；绝不把整篇 prompt / 构图段当作锚点。"""
    meta = case_meta or {}

    # 1) 显式入参 / case_meta.design_requirement —— 直接使用，不做 prompt 回退抽取
    for candidate in (
        design_requirement,
        meta.get("design_requirement"),
        meta.get("design_requirement_text"),
    ):
        text = str(candidate or "").strip()
        if not text:
            continue
        # 误把整篇合成 prompt 塞进该字段时拒绝，改走标题/意象摘要
        if "【画面构图】" in text and "【核心特征】" in text:
            continue
        return text

    parts: list[str] = []
    main = str(meta.get("main_title") or "").strip()
    sub = str(meta.get("sub_title") or "").strip()
    other = str(meta.get("other_requirements") or "").strip()
    imagery = str(meta.get("imagery_requirement") or "").strip()
    prompt = (initial_prompt or "").strip()

    if not main:
        m = _TITLE_MAIN_RE.search(prompt)
        if m:
            main = m.group(1).strip()
    if not sub:
        m = _TITLE_SUB_RE.search(prompt)
        if m:
            sub = m.group(1).strip()

    if main:
        parts.append(f"主标题：{main}")
    if sub:
        parts.append(f"副标题：{sub}")

    seen: set[str] = set()
    decors: list[str] = []
    for d in _DECOR_RE.findall(prompt):
        d = d.strip()
        if d and d not in seen and d not in (main, sub):
            seen.add(d)
            decors.append(d)
    if decors:
        parts.append("装饰文案：" + "；".join(decors[:4]))

    if imagery:
        parts.append(f"主体意象：{_clip(imagery, 180)}")
    else:
        m = _CORE_RE.search(prompt)
        if m:
            core = _clip(m.group(1), 180)
            if core:
                parts.append(f"核心特征摘要：{core}")

    if other:
        parts.append(f"其他要求：{_clip(other, 200)}")

    return "\n".join(parts)


def format_design_requirement_section(design_requirement: str) -> str:
    content = (design_requirement or "").strip()
    if not content:
        return ""
    return f"## 约束锚点（设计要求 · 不得偏离）\n{content}\n\n"
