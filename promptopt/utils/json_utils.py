"""【功能描述】LLM 响应中的 JSON 提取辅助函数。
【输入】含 JSON 的 LLM 响应文本（可含 ```json 围栏或裸 `{...}` / `[...]`）。
【输出】解析得到的 `dict` 或 `list`，解析失败时返回 `None`。
"""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | None:
    """从 LLM 响应文本中提取 JSON 对象。

    优先匹配 ```json 围栏，再尝试裸 `{...}` 模式。
    """
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
