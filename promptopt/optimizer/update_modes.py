"""【功能描述】patch 更新模式的辅助函数，统一 payload 键/标签及项描述。
【输入】update_mode 字符串、含 edits 的 container dict。
【输出】normalize_update_mode、get_payload_items、truncate_payload 等工具函数返回值。
"""
from __future__ import annotations

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
