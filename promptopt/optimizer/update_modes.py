"""【功能描述】patch 更新模式的辅助函数，统一 payload 键/标签及项描述。
【输入】update_mode 字符串、含 edits 的 container dict。
【输出】normalize_update_mode、get_payload_items、describe_item 等工具函数返回值。
"""
from __future__ import annotations

from typing import Any

PATCH_MODE = "patch"


def normalize_update_mode(mode: str | None) -> str:
    """仅支持 patch 局部编辑模式；其他值回退为 patch。"""
    raw = str(mode or PATCH_MODE).strip().lower()
    aliases = {
        "patch": PATCH_MODE,
        "edits": PATCH_MODE,
    }
    return aliases.get(raw, PATCH_MODE)


def payload_key(mode: str | None) -> str:
    return "edits"


def payload_label(mode: str | None, *, singular: bool = False, title: bool = False) -> str:
    word = "edit" if singular else "edits"
    return word.title() if title else word


def get_payload_items(container: dict | None, mode: str | None) -> list[dict]:
    if not isinstance(container, dict):
        return []
    items = container.get(payload_key(mode), [])
    return items if isinstance(items, list) else []


def set_payload_items(container: dict, items: list[dict], mode: str | None) -> dict:
    container[payload_key(mode)] = items
    return container


def truncate_payload(container: dict, max_items: int, mode: str | None) -> dict:
    if max_items < 0:
        return container
    items = get_payload_items(container, mode)
    if len(items) > max_items:
        set_payload_items(container, items[:max_items], mode)
    return container


def describe_item(item: dict, mode: str | None, *, max_chars: int = 240) -> str:
    if not isinstance(item, dict):
        return ""
    op = item.get("op", "?")
    target = item.get("target", "")
    content = item.get("content", "")
    parts = [f"op={op}"]
    if target:
        parts.append(f"target={target!r}")
    if content:
        parts.append(f"content={content!r}")
    if item.get("support_count") is not None:
        parts.append(f"support={item.get('support_count')}")
    text = "  ".join(parts)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def short_item_summary(item: dict, mode: str | None, *, max_chars: int = 200) -> dict[str, Any]:
    return {
        "op": item.get("op", "?"),
        "content": str(item.get("content", ""))[:max_chars],
        "target": item.get("target", ""),
    }
